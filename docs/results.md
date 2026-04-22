# Results

Reproduction of NEO and extensions (M6 CPU-KV INT8, M7 INT8 transfer).
All numbers in this document come from runs on `<GPU>` with
`<model>`; see per-table captions for exact configuration.

> **STATUS:** figures and numbers in tables below are placeholders. After
> the next GPU run, fill in every cell marked `TODO`. Figures live at
> `docs/figures/{fig6c,fig10a,headtohead_*}.pdf` — the placeholder paths
> here will resolve once those PDFs exist.

---

## 1. Reproduction of paper results

### 1.1 Figure 6c — load–latency (T4, Llama-2-7B, OpenAI summarization)

![fig6c](figures/fig6c.pdf)

| System | Peak throughput (req/s) | Per-token latency @ 0.5 req/s (s) | SLO-attainment @ 0.5 req/s |
| ------ | ----------------------- | --------------------------------- | -------------------------- |
| vLLM   | TODO                    | TODO                              | TODO                       |
| NEO    | TODO                    | TODO                              | TODO                       |

*Deviation from paper:* TODO (cite paper Table X, report |Δ|).

### 1.2 Figure 10a — throughput sensitivity (A10G, Llama-3-8B, synthetic)

![fig10a](figures/fig10a.pdf)

| Output length | Base (GPU-only) throughput (req/s) | NEO throughput (req/s) | Speedup |
| ------------- | ---------------------------------- | ---------------------- | ------- |
| 50            | TODO                               | TODO                   | TODO    |
| 100           | TODO                               | TODO                   | TODO    |
| 200           | TODO                               | TODO                   | TODO    |
| 300           | TODO                               | TODO                   | TODO    |
| 400           | TODO                               | TODO                   | TODO    |

*Deviation from paper:* TODO.

### 1.3 Head-to-head (custom plot)

![headtohead-latency](figures/headtohead_latency_osc.pdf)
![headtohead-throughput](figures/headtohead_throughput_osc.pdf)
![headtohead-p99](figures/headtohead_p99_osc.pdf)

*Generated via `scripts/plot_headtohead.py`.*

---

## 2. Extensions

### 2.1 M6 — INT8 CPU KV cache

**Setup.** NEO with the CPU-side KV tensors quantized to INT8 at
`per-token` granularity. GPU-side KV cache remains FP16; quantization
happens at H2D / D2H time.

| Metric                      | fp16-neo | int8-cpu-kv | Δ      |
| --------------------------- | -------- | ----------- | ------ |
| Throughput (Azure Code)     | TODO     | TODO        | TODO   |
| Throughput (OSC)            | TODO     | TODO        | TODO   |
| Per-token p99 (Azure Code)  | TODO     | TODO        | TODO   |
| WikiText-2 perplexity       | TODO     | TODO        | TODO   |
| HumanEval pass@1            | TODO     | TODO        | TODO   |
| CNN/DM ROUGE-1              | TODO     | TODO        | TODO   |

### 2.2 M7 — INT8 transfer path

**Setup.** KV blocks are INT8-quantized before the PCIe crossing and
dequantized on the receiving side. Both CPU and GPU storage remain
FP16. Exposed via `--int8-transfer`.

| Metric                       | fp16-neo | int8-transfer | Δ      |
| ---------------------------- | -------- | ------------- | ------ |
| Throughput (Azure Code)      | TODO     | TODO          | TODO   |
| Throughput (OSC)             | TODO     | TODO          | TODO   |
| Per-token p99 (Azure Code)   | TODO     | TODO          | TODO   |
| Transfer-time / block (µs)   | TODO     | TODO          | TODO   |
| WikiText-2 perplexity        | TODO     | TODO          | TODO   |
| HumanEval pass@1             | TODO     | TODO          | TODO   |
| CNN/DM ROUGE-1               | TODO     | TODO          | TODO   |

---

## 3. Granularity ablation (M10)

For both INT8 variants we sweep `--quant-granularity`. The hypothesis
is that `per-token` gives the best accuracy/throughput trade-off
because KV activations have per-token outliers, while `per-channel`
loses accuracy cheaply and `per-token-per-head` gives a marginal gain
at extra metadata cost.

![ablation-perplexity](figures/ablation_perplexity.pdf)

### 3.1 int8-cpu-kv

| Granularity          | Throughput (Azure Code) | Per-token p99 | WikiText-2 PPL | HumanEval pass@1 |
| -------------------- | ----------------------- | ------------- | -------------- | ---------------- |
| per-token            | TODO                    | TODO          | TODO           | TODO             |
| per-channel          | TODO                    | TODO          | TODO           | TODO             |
| per-token-per-head   | TODO                    | TODO          | TODO           | TODO             |

### 3.2 int8-transfer

| Granularity          | Throughput (Azure Code) | Per-token p99 | WikiText-2 PPL | HumanEval pass@1 |
| -------------------- | ----------------------- | ------------- | -------------- | ---------------- |
| per-token            | TODO                    | TODO          | TODO           | TODO             |
| per-channel          | TODO                    | TODO          | TODO           | TODO             |
| per-token-per-head   | TODO                    | TODO          | TODO           | TODO             |

*Source CSVs:* `benchmarks/results/ablation_summary.csv` (produced by
`scripts/run_ablation.sh`).

---

## 4. End-to-end downstream quality

| Variant          | HumanEval pass@1 | CNN/DM ROUGE-1 | CNN/DM ROUGE-L |
| ---------------- | ---------------- | -------------- | -------------- |
| vllm (fp16)      | TODO             | TODO           | TODO           |
| fp16-neo         | TODO             | TODO           | TODO           |
| int8-cpu-kv      | TODO             | TODO           | TODO           |
| int8-transfer    | TODO             | TODO           | TODO           |

Per-variant CSVs under `benchmarks/results/humaneval_*.csv` and
`benchmarks/results/rouge_*.csv`.

---

## 5. How to reproduce

```bash
# 1. Paper reproduction (fig6c + fig10a)
bash scripts/run_reproduction.sh --fig both

# 2. Head-to-head CSVs
python scripts/run_vllm_baseline.py --system vllm
python scripts/run_vllm_baseline.py --system ours
python scripts/plot_headtohead.py

# 3. Extension benchmarks (M8 + M9)
for v in fp16-neo int8-cpu-kv int8-transfer vllm; do
    python benchmarks/run_throughput.py  --variant "$v"
    python benchmarks/run_latency.py     --variant "$v" --workload azure-code --rate 1.0
    python benchmarks/run_perplexity.py  --variant "$v" --model-path <MODEL_PATH>
    python benchmarks/run_humaneval.py   --variant "$v"
    python benchmarks/run_rouge.py       --variant "$v"
done

# 4. Granularity ablation (M10)
bash scripts/run_ablation.sh --variant both
```

All results CSVs land under `benchmarks/results/`. Re-run
`scripts/plot_headtohead.py` to regenerate plots after new CSVs.

---

## 6. Known caveats

- **INT8 transfer path is currently a Python fallback**
  (`swiftllm/worker/block_swapper.py:_swap_blocks_int8`). A fused C++
  kernel is `TODO(gpu-verify)` — the Python path is correct but slow,
  so throughput numbers for `int8-transfer` are a lower bound until the
  kernel lands.
- **TTFT is approximated** in `benchmarks/run_latency.py` unless
  `--ttft-mode=stream` is set; true TTFT requires the streaming API.
- **Perplexity for NEO variants** requires a small `LlamaModel.forward`
  logits shim (see `benchmarks/run_perplexity.py:NEOForwardAdapter`).
  Until that shim lands, PPL cells for `int8-cpu-kv` / `int8-transfer`
  will stay `TODO`.
- **`int8-cpu-kv` flag spelling** is provisional; once M6 merges its
  CLI name takes precedence and `benchmarks/_common.py:resolve_variant`
  should be updated to match.

---

*Last updated:* `TODO(run-date)`
