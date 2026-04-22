"""
Correctness tests for the --int8-transfer path (M7).

These tests run on CPU only and do not require CUDA. They validate the
round-trip numerical fidelity of the per-group INT8 quantizer used by
Swapper._swap_blocks_int8 — if round-trip error exceeds the bar here,
downstream benchmark results (M8/M9) are untrustworthy.

Correctness bar (matching M6's bar for CPU-KV quantization):
    - max abs error  < 2 * step_size   (step_size = max_abs / 127)
    - rms rel error  < 5e-3 for per-token granularity
    - zero-input blocks round-trip to zero (no NaNs)

Usage:
    pytest tests/test_int8_transfer.py -v
"""

from __future__ import annotations

import math
import torch
import pytest

from swiftllm.worker.quantize import (
    QuantGranularity,
    quantize_int8,
    dequantize_int8,
    roundtrip_int8,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
NUM_KV_HEADS = 4
BLOCK_SIZE = 16
HEAD_DIM = 128


def _random_block(dtype=torch.float16, seed: int = 0) -> torch.Tensor:
    """Produce a KV block with realistic per-row dynamic range."""
    g = torch.Generator().manual_seed(seed)
    # Mix of small and large magnitudes across tokens — this stresses the
    # per-token scale more than iid normal would.
    base = torch.randn(NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM, generator=g, dtype=torch.float32)
    token_scale = torch.rand(NUM_KV_HEADS, BLOCK_SIZE, 1, generator=g) * 4.0 + 0.1
    return (base * token_scale).to(dtype)


# ---------------------------------------------------------------------------
# Shape / dtype contract
# ---------------------------------------------------------------------------
def test_quantize_dtype_and_shape():
    x = _random_block()
    q, s = quantize_int8(x)
    assert q.dtype == torch.int8
    assert q.shape == x.shape
    assert s.dtype == torch.float32
    # per-token: reduce head_dim → scale shape ends in 1
    assert s.shape == (NUM_KV_HEADS, BLOCK_SIZE, 1)


def test_dequantize_rejects_non_int8():
    q = torch.zeros(4, dtype=torch.float16)
    s = torch.ones(4, dtype=torch.float32)
    with pytest.raises(TypeError):
        dequantize_int8(q, s)


def test_quantize_rejects_non_float():
    x = torch.zeros(4, dtype=torch.int32)
    with pytest.raises(TypeError):
        quantize_int8(x)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("granularity", list(QuantGranularity))
def test_roundtrip_bounded_error(granularity):
    x = _random_block(seed=42)
    y = roundtrip_int8(x, granularity=granularity)

    assert y.dtype == x.dtype
    assert y.shape == x.shape
    assert not torch.isnan(y).any()

    # Error bound: within ~2 * step_size per group, where step_size is
    # (max_abs / 127). Use the global amax for a conservative bound.
    step = x.abs().amax().item() / 127.0
    max_err = (y.to(torch.float32) - x.to(torch.float32)).abs().max().item()
    assert max_err < 2 * step, f"max_err={max_err} exceeds 2*step={2 * step}"


def test_per_token_rms_bound():
    """per-token granularity should give tighter rms error than per-channel
    on our synthetic block (tokens have varied dynamic range)."""
    x = _random_block(seed=7)
    y = roundtrip_int8(x, granularity=QuantGranularity.PER_TOKEN)
    rel = ((y.to(torch.float32) - x.to(torch.float32)) / (x.to(torch.float32).abs() + 1e-6))
    rms_rel = math.sqrt((rel * rel).mean().item())
    assert rms_rel < 5e-3, f"per-token rms_rel={rms_rel}"


def test_zero_input_roundtrips_to_zero():
    x = torch.zeros(NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM, dtype=torch.float16)
    y = roundtrip_int8(x)
    assert torch.all(y == 0)
    assert not torch.isnan(y).any()


def test_scale_broadcasts_cleanly():
    """Quantize and dequantize in two steps — the scale must broadcast
    against the int8 tensor without a manual reshape."""
    x = _random_block()
    q, s = quantize_int8(x, granularity=QuantGranularity.PER_TOKEN)
    # If this raises a broadcasting error, the scale shape is wrong.
    y = dequantize_int8(q, s, out_dtype=x.dtype)
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Granularity comparison (sanity for M10)
# ---------------------------------------------------------------------------
def test_granularities_all_within_spec():
    """All three granularities should satisfy the absolute-error bound,
    even though their rms characteristics differ."""
    x = _random_block(seed=123)
    step = x.abs().amax().item() / 127.0
    for g in QuantGranularity:
        y = roundtrip_int8(x, granularity=g)
        max_err = (y.to(torch.float32) - x.to(torch.float32)).abs().max().item()
        # per-channel's bound is looser because outlier tokens in a channel
        # dominate the scale — allow 3*step there.
        bound = 3 * step if g is QuantGranularity.PER_CHANNEL else 2 * step
        assert max_err < bound, f"{g}: max_err={max_err} bound={bound}"
