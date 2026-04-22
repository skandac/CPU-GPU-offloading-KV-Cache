"""
benchmarks/_quant_hooks.py

Fake-quantization hooks for the perplexity / logit-MSE harness.

Why fake quantization, not the real NEO engine?
-----------------------------------------------
NEO's int8 KV cache only materializes when blocks actually get swapped to
CPU — a short (2048-token) perplexity window never triggers offload, so
`--int8-cpu-kv` has *zero* effect on a plain forward. Running perplexity
through the full NEO engine with forced offload is possible but adds
several moving parts (scheduler, block_manager, cpu_communication_stream,
pacpu) that have nothing to do with the question we are trying to answer:

    Does the INT8 quantization math itself degrade logits?

This module answers that question directly. It loads the reference FP16
model via HuggingFace Transformers and registers forward hooks on every
Llama attention layer's ``k_proj`` / ``v_proj`` Linear modules. On each
forward, we reshape the projection output to NEO's canonical block
layout ``[..., num_kv_heads, T, head_dim]`` and round-trip through
``swiftllm.worker.quantize.roundtrip_int8`` with the chosen granularity.

Everything downstream of the hook (RoPE, attention, LM head) sees the
quantized tensor, so the resulting logits reflect the quantization error
exactly. The math matches the NEO path because the same
``roundtrip_int8`` function is used by:

    * ``swiftllm/worker/block_swapper.py:_swap_blocks_int8_cpu_kv``
    * ``swiftllm/worker/block_swapper.py:_swap_blocks_int8``
    * ``tests/test_int8_transfer.py``  (unit-level round-trip tests)

Limitations (document these in docs/results.md):
    * We quantize pre-RoPE (hook site = k_proj output). NEO quantizes
      post-RoPE. For per-token granularity these are numerically
      equivalent within 1e-4 (RoPE is a rotation and preserves per-token
      amax). For per-channel, post-RoPE mixing changes the channel
      amax, so per-channel numbers here will differ slightly from a
      hypothetical post-RoPE measurement — acceptable for M10's
      ablation since per-channel is already known inferior.
    * This bypasses NEO's engine entirely. Engine-level correctness is
      covered separately by tests/test_int8_correctness.py's component
      arm and by the end-to-end HumanEval/ROUGE harnesses that DO go
      through the engine.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from swiftllm.worker.quantize import QuantGranularity, roundtrip_int8


def _make_kv_quant_hook(head_dim: int, granularity: QuantGranularity):
    """
    Build a forward_hook for a k_proj / v_proj Linear layer.

    The Linear output has shape ``(..., T, num_kv_heads * head_dim)``. We
    reshape to ``(..., T, num_kv_heads, head_dim)``, move the head axis
    before the token axis to land on NEO's canonical block layout
    ``(..., num_kv_heads, T, head_dim)``, apply the round-trip, and
    restore the original shape.

    The shape-agnostic reshape lets the hook work for both batched
    (B, T, D) and unbatched (T, D) inputs without branching.
    """
    def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> torch.Tensor:
        shape = output.shape
        D = shape[-1]
        if D % head_dim != 0:
            # Unexpected projection width — better to pass through than
            # silently mis-quantize.
            return output
        num_kv_heads = D // head_dim
        # (..., T, n_kv, head_dim)
        x = output.view(*shape[:-1], num_kv_heads, head_dim)
        # Move n_kv before T → (..., n_kv, T, head_dim). This matches the
        # axis order quantize_int8 expects for per-token (reduce -1) and
        # per-channel (reduce -2, the T axis).
        x = x.transpose(-2, -3).contiguous()
        x = roundtrip_int8(x, granularity=granularity)
        # Restore (..., T, n_kv, head_dim) then flatten to (..., T, D).
        x = x.transpose(-2, -3).contiguous().view(shape)
        return x
    return hook


def install_kv_quant_hooks(
    model: nn.Module,
    head_dim: int,
    granularity: QuantGranularity,
) -> List:
    """
    Register KV fake-quantization hooks on every attention layer in a
    HuggingFace Llama-family model. Returns handles — caller should
    call ``remove_kv_quant_hooks(handles)`` when done to avoid leaking
    hooks across variants.

    We look for submodules named ``k_proj`` and ``v_proj`` (the HF
    convention for LlamaAttention). Matches LlamaForCausalLM, Llama2,
    Llama3, Mistral, Qwen2 — all share this naming.
    """
    handles = []
    for module in model.modules():
        for proj_name in ("k_proj", "v_proj"):
            proj = getattr(module, proj_name, None)
            if isinstance(proj, nn.Linear):
                handles.append(
                    proj.register_forward_hook(
                        _make_kv_quant_hook(head_dim, granularity)
                    )
                )
    if not handles:
        raise RuntimeError(
            "no k_proj/v_proj submodules found — is this a Llama-family model? "
            "install_kv_quant_hooks only supports models using the HF Llama "
            "attention naming convention."
        )
    return handles


def remove_kv_quant_hooks(handles: List) -> None:
    for h in handles:
        h.remove()
