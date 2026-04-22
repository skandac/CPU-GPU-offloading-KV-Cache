# INT8 KV-cache quantization design

Design doc for M6 (INT8 on CPU KV cache) and, transitively, M7
(INT8 on the transfer path) and M10 (granularity ablation). The goal
is to halve CPU KV footprint and halve H2D/D2H bytes without
measurable generation-quality loss.

**Decision summary** — the rest of the doc justifies each of these:

| Choice | Value |
| ------ | ----- |
| Scheme                  | Symmetric INT8 |
| Granularity             | Per-token (one scale per `(layer, block, kv_head, token)`) |
| Scale dtype             | FP16 |
| Layout                  | Scales live alongside blocks, in a parallel tensor |
| Dequant location        | Fused inside pacpu's ISPC loop |
| Flag                    | `--int8-cpu-kv` |
| Default granularity     | `per-token` (overridable via `--quant-granularity`) |

---

## 1. Why quantize the CPU cache at all

The CPU KV pool (`k_swap`/`v_swap` in `block_swapper.py`) is the largest
host-memory allocation in NEO. At Llama-3-8B / block_size=16 /
head_dim=128 / num_kv_heads=8, one block is

    2 (k+v) × 8 heads × 16 tokens × 128 dim × 2 B = 64 KB per block per layer.

With 32 layers and 100k CPU blocks, that's ~200 GB. Halving that to 100 GB
(INT8 + ~1-2% scale overhead) lets the same box hold 2× the offloaded
sequences, which is what unlocks the pipelined mode at higher offered
rates (see [kv-offload-path.md](kv-offload-path.md), §2).

Separately, the same INT8 representation can be used on the wire
during H2D/D2H (M7), halving PCIe bytes for swap-in / swap-out.

---

## 2. Symmetric vs asymmetric

**Asymmetric** uses `(scale, zero_point)` per group:
  `q = round((x - zero_point) / scale)`, values in `[0, 255]`.

**Symmetric** uses `scale` only:
  `q = round(x / scale)`, values in `[-127, 127]` (−128 reserved to keep
  the range symmetric and the scale computation division-free).

Two reasons to pick **symmetric** here:

1. **KV activations are roughly zero-centered** after RMSNorm and the
   attention projection, so a zero-point gives marginal precision gain
   at the cost of an extra int16 per group.
2. **Fused dequant is cheaper** — `x ≈ q * scale` is a single FMA in the
   ISPC kernel; asymmetric needs `(q - zp) * scale`, branching on zp
   alignment. pacpu's hot loop is memory-bound, so every cycle counts.

KIVI (arxiv 2402.02750, §4.2) also reports the gap between symmetric
and asymmetric is <0.1 perplexity at INT8 on KV cache — below our test
noise floor.

---

## 3. Granularity

KV activations exhibit **per-token outliers**: a small fraction of
tokens has much larger magnitudes than the rest of the sequence
(KVQuant §3, arxiv 2401.18079). If we use one scale per channel, those
outliers dominate the channel max-abs and crush every other token's
INT8 resolution. The fix is to collapse the scale along the *head_dim*
axis (per token), so each token gets its own scale.

| Granularity          | Scale shape per block          | Scales / block | Error characteristic |
| -------------------- | ------------------------------ | -------------- | -------------------- |
| per-channel          | `(num_kv_heads, 1, head_dim)`  | 1024           | Outliers poison channels |
| per-token            | `(num_kv_heads, block_size, 1)` | 128           | Outliers confined to their token |
| per-token-per-head   | same as per-token (equivalent in this layout) | 128 | No gain here |

Per-token-per-head is called out in M10 for parity with KVQuant's naming,
but in NEO's `(num_kv_heads, block_size, head_dim)` block layout the
head axis is already separate, so per-token reduces only `head_dim` —
which is already per-token-per-head. We ship it as a distinct enum
value anyway so ablation tables can distinguish "we meant it" from
"we fell through to per-token."

### Granularity × code path (M10 scope)

| Granularity          | `--int8-transfer` (M7) | `--int8-cpu-kv` (M6) |
| -------------------- | ---------------------- | -------------------- |
| per-token            | ✅ supported            | ✅ supported          |
| per-token-per-head   | ✅ supported            | ✅ supported (≡ per-token in this layout) |
| per-channel          | ✅ supported            | ❌ not supported — guard in `block_swapper.__init__` raises |

Why M6 can't do per-channel: the online append path writes one new
token per step via `brute::store_kv_int8`. Per-channel scales span all
`block_size` tokens in the block, so every appended token would
invalidate the scale for the entire channel and force re-quantization
of the whole block. That's infeasible on the hot decode path, so we
fail-fast when `--int8-cpu-kv --quant-granularity=per-channel` is
combined. The per-channel ablation cell is still reachable through
`--int8-transfer`, where no persistent scale tensor is stored.

**Default: per-token.** This matches KIVI's recommended KV scheme and
is what most recent open-source int8 KV implementations (llama.cpp Q8_0,
vLLM's experimental int8 KV path) use.

---

## 4. Scale storage layout

Two realistic options:

### Option A — interleaved (scales inside the block)
Extend each block by 16 bytes of scales, so one block is
`(num_kv_heads, block_size, head_dim + ε)`. Pro: single DMA, great locality.
Con: breaks the clean `head_dim`-aligned layout that paged-attention
kernels assume; breaks the GPU kernel for no reason (GPU stays FP16).

### Option B — parallel scale tensor (chosen)
Separate tensor `k_swap_scales` of shape
`(num_layers, num_cpu_blocks, num_kv_heads, block_size, 1)`, dtype FP16.

Pros:
- No change to the block's physical layout — the paged-attention
  kernel reads scales with a second pointer, nothing more.
- GPU-side FP16 cache is untouched (M6 requirement).
- H2D/D2H for M7 gains a second small copy, but the scale tensor is
  ~1/128th the size of the block for per-token granularity, so PCIe
  cost is negligible.

Size overhead per block: `8 × 16 × 2 B = 256 B` — 0.4% of a 64 KB block.

### Scale dtype
FP16 is sufficient for the scales because:
- The scale dynamic range is bounded by `amax(x) / 127` where `x` is
  already FP16.
- A scale overflow would require `amax(x) > 65504 × 127 ≈ 8.3e6`, which
  is physically impossible for post-RMSNorm KV activations.

FP32 scales would add 256 B more per block for zero measurable accuracy
benefit.

---

## 5. Dequant location

Three places dequantization could live:

1. **After copy to GPU, before attention** — materialize the FP16 block
   on GPU, then feed it to paged-attention as today. Easiest, but wastes
   GPU memory for a block that will be consumed once.
2. **Inside the GPU paged-attention kernel** — fused dequant. Fastest on
   GPU, but touches the most complex part of the codebase.
3. **Inside the CPU paged-attention kernel (pacpu)** — fused dequant on
   CPU. Cheapest change scope, and the CPU side is where the int8 cache
   actually lives; swap-in to GPU already triggers a dequantize step to
   restore FP16 (see M7's transfer path), so the GPU kernel stays
   untouched.

**Choice: option 3.** The GPU KV cache remains FP16 (explicit M6
constraint). On swap-in we dequantize on the GPU as the last step
before writing into `k_cache`/`v_cache` — so after swap-in, the GPU
sees only FP16 and the GPU attention kernel is unchanged.

The pacpu kernel accepts two extra pointers (`k_cache_scales`,
`v_cache_scales`) and does the FMA `x = q * s` inside its inner ISPC
loop. The scales are broadcast over `head_dim`, so per-token granularity
adds exactly one FP16 load per token per head per block — amortized
across all of `head_dim` FMAs, this is lost in the noise.

---

## 6. Reference points

- **KIVI** (Liu et al., ICML 2024, arxiv 2402.02750) — **per-channel key /
  per-token value** at 2-bit. Their per-token-value rule confirms that
  V activations have per-token outliers. At INT8 the granularity gap
  collapses; we use per-token for both to keep one code path.
- **KVQuant** (Hooper et al., NeurIPS 2024 spotlight, arxiv 2401.18079) —
  per-token-per-head quantization + offline outlier isolation. Their
  finding that the outlier set is small (<1%) is what lets symmetric
  INT8 work without zero-points for our setting.
- **vLLM INT8 KV experimental** (vllm-project/vllm #2543) — per-tensor
  scales, reports 0.02 PPL loss on WikiText-2 at INT8. We expect
  per-token to be tighter, <0.01 PPL.
- **llama.cpp Q8_0** — block-of-32 symmetric INT8 with FP16 scale. Our
  per-token is a superset: `block_size=16` per-token-per-kv-head is
  essentially Q8_0 with slightly coarser blocks.

---

## 7. Acceptance bar

For M6 to be considered correct:

- **Logit MSE < 1e-3** between FP16 and INT8 paths on 10 fixed prompts
  (see `tests/test_int8_correctness.py`).
- **WikiText-2 perplexity delta < 0.05** on Llama-2-7B and Llama-3-8B
  (reported in `docs/results.md`, M11).
- **No change** to the FP16 throughput number — i.e. when
  `--int8-cpu-kv` is off, performance exactly matches today.
- **HumanEval pass@1 delta ≤ 1 problem** (0.6% absolute) — matching
  the bar KVQuant uses for their "lossless" threshold.

Numbers above this bar are a regression and block the milestone.

---

## 8. What's out of scope here

- 4-bit / 2-bit quantization (would need asymmetric + outlier isolation
  à la KIVI/KVQuant — not worth the complexity at this stage).
- Quantizing activations or weights (unrelated — the M4 doc scopes this
  to KV).
- GPU KV cache quantization (explicit M6 constraint: GPU path stays FP16).
- Dynamic (calibration-based) scale computation — we use online max-abs
  per group, not a calibrated distribution fit.

---

*Next:* M6 implementation lives under `swiftllm/worker/block_swapper.py`,
`pacpu/pacpu.cpp`, and `pacpu/pacpu.ispc`; the new test is
`tests/test_int8_correctness.py`.
