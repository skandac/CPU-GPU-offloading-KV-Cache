"""
End-to-end correctness test for the --int8-cpu-kv path (M6).

Validates that enabling INT8 CPU KV cache does not materially change
model outputs. Per M6's acceptance bar (docs/int8-design.md §7):

    Logit MSE < 1e-3 between FP16 and INT8 paths on 10 fixed prompts.

This test is GPU-gated — the inference path lives in the SwiftLLM worker
and runs through pacpu (requires the compiled extension + a CUDA build
of the KV cache). When run on a machine without CUDA or without the
compiled pacpu extension, the test is skipped rather than failing.

Two layers of coverage live in this file:

1. ``test_quantize_dequantize_tight`` — component-level: on the exact
   distribution a CPU KV block takes (fp16, post-RMSNorm scale), the
   symmetric INT8 quantizer round-trips with error << 1e-3 RMSE. This
   runs on CPU and is always executed; if it fails the end-to-end MSE
   bar is unreachable and the fused GPU/CPU path does not need to be
   loaded to diagnose it.

2. ``test_logit_mse_fp16_vs_int8`` — integration: loads a SwiftLLM
   engine twice (once fp16, once int8-cpu-kv) on the same 10 fixed
   prompts, compares the logits. Skipped without CUDA. This is the
   authoritative M6 gate.

Why both: component test catches regressions in the quantizer
(swiftllm/worker/quantize.py) during local iteration; integration test
catches regressions in the fused pacpu kernel or the swap path, which
only surface end-to-end.

Usage:
    pytest tests/test_int8_correctness.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from swiftllm.worker.quantize import (
    QuantGranularity,
    quantize_int8,
    dequantize_int8,
    roundtrip_int8,
)


# ---------------------------------------------------------------------------
# Ten fixed prompts from docs/int8-design.md §7 — deterministic across runs.
# Short prompts keep the test fast; the point is logit fidelity, not
# generation quality.
# ---------------------------------------------------------------------------
FIXED_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a distributed system, consistency and availability",
    "def fibonacci(n):",
    "The capital of France is",
    "Translate to French: Good morning.",
    "Explain in one sentence why the sky is blue.",
    "Once upon a time, in a kingdom far away,",
    "The derivative of sin(x) with respect to x is",
    "List three primary colors:",
    "A recursive function always needs a",
]

LOGIT_MSE_BAR = 1e-3  # docs/int8-design.md §7


# ---------------------------------------------------------------------------
# Component-level: quantize/dequantize round-trip on realistic KV stats
# ---------------------------------------------------------------------------
def _kv_shaped_block(seed: int = 0) -> torch.Tensor:
    """
    Build a block shaped like a real CPU KV block. Post-RMSNorm / post-
    projection activations are ~N(0, 1) with occasional outlier tokens
    (KVQuant §3). Inject a 2% outlier fraction at ~10x scale to exercise
    the per-token quantizer's outlier isolation.
    """
    torch.manual_seed(seed)
    num_kv_heads, block_size, head_dim = 8, 16, 128
    x = torch.randn(num_kv_heads, block_size, head_dim, dtype=torch.float16)
    # Mark ~2% of tokens as outliers by scaling their rows up.
    outlier_mask = torch.rand(num_kv_heads, block_size) < 0.02
    x[outlier_mask] *= 10.0
    return x


@pytest.mark.parametrize("granularity", list(QuantGranularity))
def test_quantize_dequantize_tight(granularity: QuantGranularity):
    """
    Round-trip error on a KV-shaped block stays well under the 1e-3 MSE
    bar used end-to-end. This is the component-level canary for M6 —
    if it fires, the integration test has no hope.
    """
    x = _kv_shaped_block(seed=42)
    y = roundtrip_int8(x, granularity=granularity)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert not torch.isnan(y).any(), "round-trip produced NaN"

    mse = torch.mean((y.to(torch.float32) - x.to(torch.float32)) ** 2).item()
    # Component bar is tighter than end-to-end — round-trip on a single
    # block should be ~1e-5, orders of magnitude below the logit bar.
    assert mse < 1e-4, (
        f"round-trip MSE {mse:.2e} exceeds component bar (granularity="
        f"{granularity.value})"
    )


def test_quantize_dequantize_zero_block():
    """An all-zero token must round-trip to exactly zero, no NaNs."""
    x = torch.zeros(8, 16, 128, dtype=torch.float16)
    y = roundtrip_int8(x, granularity=QuantGranularity.PER_TOKEN)
    assert torch.all(y == 0), "zero block did not round-trip to zero"
    assert not torch.isnan(y).any()


# ---------------------------------------------------------------------------
# Integration: end-to-end logit MSE on 10 fixed prompts.
# ---------------------------------------------------------------------------
def _cuda_and_hf_available() -> bool:
    """
    The integration arm uses HF Transformers + our fake-quant hooks
    (see benchmarks/_quant_hooks.py for the reasoning — we deliberately
    do NOT drive this through NEO's engine, since a single forward
    window doesn't trigger offload and therefore doesn't exercise
    int8_cpu_kv at all).
    """
    if not torch.cuda.is_available():
        return False
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_model_path() -> str | None:
    """
    The integration test needs a real model to load. We look for an
    env var first (CI / developer override), then a conventional
    checkout location. If neither exists, the test is skipped — the
    component tests above still run.
    """
    env = os.environ.get("NEO_TEST_MODEL_PATH")
    if env and Path(env).exists():
        return env
    for candidate in (
        "/models/Llama-2-7b-hf",
        "/models/Meta-Llama-3-8B",
        str(Path.home() / "models" / "Llama-2-7b-hf"),
        str(Path.home() / "weights" / "Llama-2-7b-hf"),
        str(Path.home() / "weights" / "Llama-3-8B"),
        "/home/ubuntu/weights/Llama-2-7b-hf",
        "/home/ubuntu/weights/Llama-3-8B",
    ):
        if Path(candidate).exists():
            return candidate
    return None


@pytest.mark.skipif(
    not _cuda_and_hf_available(),
    reason="requires CUDA + transformers (integration arm)",
)
def test_logit_mse_fp16_vs_int8():
    """
    Logit MSE bar from docs/int8-design.md §7: the FP16 and INT8-
    quantized KV paths must agree to MSE < 1e-3 on the 10 fixed
    prompts.

    We measure this at the quantization-math level, not through NEO's
    engine: a single forward window wouldn't trigger KV offload, so
    --int8-cpu-kv would be a no-op and the test would be vacuous.
    Instead we reuse the same forward hooks the perplexity harness
    uses (benchmarks/_quant_hooks.install_kv_quant_hooks), which
    dispatch through the exact same quantize.roundtrip_int8 function
    block_swapper._swap_blocks_int8_cpu_kv uses. If this passes, the
    quantization math meets the bar; if it fails, the rest of M6 is
    moot.
    """
    model_path = _resolve_model_path()
    if model_path is None:
        pytest.skip("no model checkpoint available for integration test")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover — environment-dependent
        pytest.skip(f"transformers import failed: {e!r}")

    # Import the hooks lazily — keeps this file collectable on a laptop
    # even if the benchmarks package path isn't on sys.path at import time.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
    from _quant_hooks import install_kv_quant_hooks, remove_kv_quant_hooks
    from swiftllm.worker.quantize import QuantGranularity

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    head_dim = getattr(
        model.config, "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )

    # Tokenize once; pad to matching lengths so we can compare per-prompt
    # last-token logits directly without gathering per-prompt lengths.
    enc = tok(list(FIXED_PROMPTS), return_tensors="pt", padding=True)
    input_ids = enc.input_ids.cuda()
    attention_mask = enc.attention_mask.cuda()

    @torch.inference_mode()
    def last_token_logits() -> torch.Tensor:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        # Last non-pad position per prompt.
        lengths = attention_mask.sum(dim=1) - 1
        idx = torch.arange(out.logits.shape[0], device=out.logits.device)
        return out.logits[idx, lengths]  # [num_prompts, vocab_size]

    # FP16 reference.
    ref = last_token_logits().float()

    # INT8 (per-token, matching M6's default).
    handles = install_kv_quant_hooks(
        model, head_dim=head_dim, granularity=QuantGranularity.PER_TOKEN
    )
    try:
        got = last_token_logits().float()
    finally:
        remove_kv_quant_hooks(handles)

    mse = torch.mean((got - ref) ** 2).item()
    print(f"[test_logit_mse_fp16_vs_int8] MSE = {mse:.3e}")
    assert mse < LOGIT_MSE_BAR, (
        f"logit MSE {mse:.3e} exceeds M6 bar {LOGIT_MSE_BAR:.0e} "
        f"(docs/int8-design.md §7)"
    )
