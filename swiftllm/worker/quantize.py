"""
INT8 quantization helpers for the KV-cache transfer path.

Scope
-----
These helpers are used when the engine is launched with ``--int8-transfer``.
In that mode, KV blocks are compressed to INT8 before the H2D / D2H copy
and decompressed back to FP16 on the receiving side. CPU- and GPU-resident
KV storage stays FP16; only the bytes on the wire are INT8.

This is *per-token* granularity by default: one scale per
``(layer, block, kv_head, token)`` slot — the scale tensor has the same
shape as the block but with ``head_dim`` collapsed to 1. That matches the
layout Q8 attention papers use and keeps compression simple (no zero
point, symmetric range).

For M10 the granularity is configurable — see ``QuantGranularity``.

All functions are pure torch and run on CPU or CUDA. They are written so
tests can exercise them without a GPU.

TODO(gpu-verify): on CUDA, ``.amax`` over the head_dim axis will be
memory-bound. If profiling shows it's the bottleneck, rewrite with a
fused triton kernel.
"""

from __future__ import annotations

import enum
from typing import Tuple

import torch

# INT8 symmetric range. -128 is reserved to keep scale computation
# symmetric; we map [-127, 127] <-> [-max_abs, +max_abs].
_INT8_MAX = 127.0


class QuantGranularity(str, enum.Enum):
    """
    Granularity axes for INT8 scale computation.

    - PER_TOKEN:          one scale per (layer, block, kv_head, token)
    - PER_CHANNEL:        one scale per (layer, block, kv_head, channel)
    - PER_TOKEN_PER_HEAD: one scale per (layer, block, kv_head, token) but
                          sharing across head_dim channels within the head
                          — identical to PER_TOKEN for the KV layout used
                          here, retained as a distinct enum value for
                          clarity in ablation tables.
    """
    PER_TOKEN = "per-token"
    PER_CHANNEL = "per-channel"
    PER_TOKEN_PER_HEAD = "per-token-per-head"

    @classmethod
    def reduce_dims(cls, g: "QuantGranularity") -> Tuple[int, ...]:
        """
        Return the dims to reduce over when computing the per-group max.
        Input block shape is (..., kv_head, token, head_dim); the last dim
        is always either reduced or kept.
        """
        if g in (cls.PER_TOKEN, cls.PER_TOKEN_PER_HEAD):
            return (-1,)   # reduce head_dim, keep one scale per token
        if g is cls.PER_CHANNEL:
            return (-2,)   # reduce token axis, keep one scale per channel
        raise ValueError(f"unknown granularity: {g}")


def quantize_int8(
    x: torch.Tensor,
    granularity: QuantGranularity = QuantGranularity.PER_TOKEN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize ``x`` (fp16 / fp32) to INT8 with per-group symmetric scales.

    Parameters
    ----------
    x : torch.Tensor
        Any floating tensor. Typical shape for KV blocks is
        ``(num_kv_heads, block_size, head_dim)``.
    granularity : QuantGranularity
        Controls which axis the scale collapses along.

    Returns
    -------
    q : torch.Tensor (int8)
        Same shape as ``x``.
    scale : torch.Tensor (float32)
        Same rank as ``x`` with the reduced dim kept at size 1 so it
        broadcasts cleanly during dequantization.
    """
    if not x.is_floating_point():
        raise TypeError(f"expected floating input, got {x.dtype}")

    dims = QuantGranularity.reduce_dims(granularity)
    # max-abs in fp32 to avoid fp16 overflow on large blocks
    amax = x.detach().abs().amax(dim=dims, keepdim=True).to(torch.float32)
    # clamp to avoid division by zero on all-zero groups
    scale = torch.clamp(amax, min=1e-6) / _INT8_MAX
    q = torch.round(x.to(torch.float32) / scale).clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    return q, scale


def dequantize_int8(
    q: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Inverse of :func:`quantize_int8`.

    Parameters
    ----------
    q : torch.Tensor (int8)
    scale : torch.Tensor
        Must broadcast against ``q``.
    out_dtype : torch.dtype
        Destination floating dtype (fp16 by default, matching KV storage).
    """
    if q.dtype != torch.int8:
        raise TypeError(f"expected int8 input, got {q.dtype}")
    return (q.to(torch.float32) * scale).to(out_dtype)


# ---------------------------------------------------------------------------
# Transfer-path helpers
# ---------------------------------------------------------------------------
def roundtrip_int8(
    x: torch.Tensor,
    granularity: QuantGranularity = QuantGranularity.PER_TOKEN,
) -> torch.Tensor:
    """
    Convenience: quantize then dequantize back to ``x.dtype``. Used as the
    Python fallback inside block_swapper when --int8-transfer is on and
    the fused C++ int8 swap kernel is unavailable.

    This materializes the int8 tensor, so callers that want to cross a
    device boundary should call :func:`quantize_int8` on the source side
    and :func:`dequantize_int8` on the destination side instead — that
    way only int8 bytes cross the PCIe bus.
    """
    q, s = quantize_int8(x, granularity=granularity)
    return dequantize_int8(q, s, out_dtype=x.dtype)
