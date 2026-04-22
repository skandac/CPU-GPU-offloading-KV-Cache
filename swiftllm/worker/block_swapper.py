"""
BlockManager and Swapper

Contains initalization and transition logics for the KV cache.
"""

import torch
import swiftllm_c
from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.quantize import (
    QuantGranularity,
    quantize_int8,
    dequantize_int8,
)

class Swapper:
    """
    Swapper - Manage the swapping of sequences in and out of the model

    This manager is responsible for swapping sequences in and out of the model.
    It maintains the block manager, and provides methods to swap sequences in
    and out.
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: LlamaModelConfig
    ):
        self.engine_config = engine_config
        self.model_config = model_config

        num_q_heads = model_config.num_q_heads // model_config.world_size
        num_kv_heads = model_config.num_kv_heads // model_config.world_size

        # Initialize KV cache
        kvcache_shape = (
            model_config.num_layers + engine_config.extra_layer_for_cprf,
            engine_config.num_gpu_blocks,
            num_kv_heads,
            engine_config.block_size,
            model_config.head_dim
        )
        # Here we use torch.zeros instead of torch.empty, since that torch.empty
        # has the possibility to contain NaNs, which will cause the model to output NaNs.
        self.k_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")
        self.v_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")

        # Initialize KV swap space.
        # When int8-cpu-kv is enabled, the swap tensors are INT8 and a parallel
        # scale tensor holds one FP16 scale per (layer, block, head, token).
        # See docs/int8-design.md §4 for the layout rationale.
        kvswap_shape = (
            model_config.num_layers,
            engine_config.num_cpu_blocks,
            num_kv_heads,
            engine_config.block_size,
            model_config.head_dim
        )
        swap_dtype = torch.int8 if engine_config.int8_cpu_kv else torch.float16
        self.k_swap = torch.zeros(kvswap_shape, dtype=swap_dtype, device="cpu", pin_memory=True)
        self.v_swap = torch.zeros(kvswap_shape, dtype=swap_dtype, device="cpu", pin_memory=True)

        # Parallel FP16 scale tensors — only allocated when int8_cpu_kv is on.
        # The CPU KV scale layout is fixed at (num_kv_heads, block_size, 1) —
        # one scale per (head, token). This matches the pacpu kernel's
        # fused-dequant expectation (docs/int8-design.md §5) and the on-
        # the-fly append path (`brute::store_kv_int8`), which can only
        # honor per-token granularity: re-quantizing an entire block on
        # every appended token (what per-channel would require) is not
        # viable on the hot decode path.
        #
        # M10 ablation scope: per-token and per-token-per-head are
        # equivalent in this block layout (§3). Per-channel is not
        # supported with --int8-cpu-kv — the scale tensor layout would
        # have to change and the append-path kernel would have to
        # re-quantize every block per step. Per-channel remains
        # exercisable through --int8-transfer, where no persistent
        # scale tensor is stored.
        if engine_config.int8_cpu_kv:
            if engine_config.quant_granularity == "per-channel":
                raise ValueError(
                    "--quant-granularity=per-channel is incompatible with "
                    "--int8-cpu-kv; the online append path cannot honor "
                    "per-channel without re-quantizing the whole block on "
                    "every token. Use per-token or per-token-per-head, or "
                    "use --int8-transfer for the per-channel ablation. "
                    "See docs/int8-design.md §3, §5."
                )
            kvscale_shape = (
                model_config.num_layers,
                engine_config.num_cpu_blocks,
                num_kv_heads,
                engine_config.block_size,
                1,
            )
            self.k_swap_scales = torch.zeros(kvscale_shape, dtype=torch.float16, device="cpu", pin_memory=True)
            self.v_swap_scales = torch.zeros(kvscale_shape, dtype=torch.float16, device="cpu", pin_memory=True)
        else:
            self.k_swap_scales = None
            self.v_swap_scales = None

        # Initialize CPU QKV buffer
        qo_cpu_shape = (engine_config.max_batch_size, num_q_heads, model_config.head_dim)
        kv_cpu_shape = (engine_config.max_batch_size, num_kv_heads, model_config.head_dim)
        self.q_cpu = torch.zeros(qo_cpu_shape, dtype=torch.float16, device="cpu", pin_memory=True)
        self.k_cpu = torch.zeros(kv_cpu_shape, dtype=torch.float16, device="cpu", pin_memory=True)
        self.v_cpu = torch.zeros(kv_cpu_shape, dtype=torch.float16, device="cpu", pin_memory=True)
        # We store float32 tensors for the output, but convert them to float16 after copying back to GPU
        self.o_cpu = torch.zeros(qo_cpu_shape, dtype=torch.float32, device="cpu", pin_memory=True)

        self.gpu_block_table = torch.zeros(
            (engine_config.max_seqs_in_block_table, engine_config.max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda"
        )
        self.cpu_block_table = torch.zeros(
            (engine_config.max_seqs_in_block_table, engine_config.max_blocks_per_seq),
            dtype=torch.int32,
            device="cpu"
        )

    
    def swap_blocks(
        self,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        is_swap_out: bool,
        gpu_layer: int,
        cpu_layer: int
    ):
        """
        Swap blocks between the GPU and CPU, the physical indexes of the blocks are given.
        """
        # pylint: disable=too-many-arguments, c-extension-no-member
        assert len(src_block_ids) == len(dst_block_ids), "Length mismatch between src_block_ids and dst_block_ids"
        if not src_block_ids:
            return

        if self.engine_config.int8_cpu_kv:
            # The CPU cache itself is INT8; swap-out must quantize before the
            # copy, swap-in must dequantize after. Subsumes --int8-transfer
            # because the wire bytes are already INT8.
            self._swap_blocks_int8_cpu_kv(
                src_block_ids, dst_block_ids, is_swap_out, gpu_layer, cpu_layer
            )
            return

        if self.engine_config.int8_transfer:
            self._swap_blocks_int8(
                src_block_ids, dst_block_ids, is_swap_out, gpu_layer, cpu_layer
            )
            return

        swiftllm_c.swap_blocks(
            src_block_ids,
            dst_block_ids,
            is_swap_out,
            gpu_layer,
            cpu_layer,

            self.k_cache, self.v_cache,
            self.k_swap, self.v_swap
        )

    def _swap_blocks_int8_cpu_kv(
        self,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        is_swap_out: bool,
        gpu_layer: int,
        cpu_layer: int,
    ):
        """
        Swap between FP16 GPU cache and INT8 CPU cache (M6).

        Direction semantics:
            is_swap_out=True  :  FP16 GPU block  →  quantize → INT8 CPU block + FP16 scale
            is_swap_out=False :  INT8 CPU block + FP16 scale  →  dequantize → FP16 GPU block

        On swap-out we quantize *on the GPU* before crossing the bus so only
        the INT8 bytes + scale transit (the same bandwidth saving as M7).
        On swap-in we copy INT8 + scale to GPU first and dequantize there.

        GPU kernel path is untouched: after swap-in, the GPU's k_cache / v_cache
        sees FP16 and runs the standard paged-attention kernel unchanged.

        TODO(gpu-verify): replace the Python loop with a fused
        ``swiftllm_c.swap_blocks_int8_cpu_kv`` kernel once it's available.
        This loop is correct but slow.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        granularity = QuantGranularity(self.engine_config.quant_granularity)

        if is_swap_out:
            # FP16 GPU → INT8 CPU + scale
            for src_pid, dst_pid in zip(src_block_ids, dst_block_ids):
                k_block = self.k_cache[gpu_layer, src_pid]   # fp16, on GPU
                v_block = self.v_cache[gpu_layer, src_pid]

                qk, sk = quantize_int8(k_block, granularity=granularity)
                qv, sv = quantize_int8(v_block, granularity=granularity)

                # Scales: fp32 from quantize_int8 — cast to fp16 for storage.
                # See docs/int8-design.md §4 for why fp16 suffices.
                self.k_swap[cpu_layer, dst_pid] = qk.to("cpu", non_blocking=True)
                self.v_swap[cpu_layer, dst_pid] = qv.to("cpu", non_blocking=True)
                self.k_swap_scales[cpu_layer, dst_pid] = sk.to(dtype=torch.float16).to("cpu", non_blocking=True)
                self.v_swap_scales[cpu_layer, dst_pid] = sv.to(dtype=torch.float16).to("cpu", non_blocking=True)
        else:
            # INT8 CPU + scale → FP16 GPU
            for src_pid, dst_pid in zip(src_block_ids, dst_block_ids):
                qk = self.k_swap[cpu_layer, src_pid].to("cuda", non_blocking=True)
                qv = self.v_swap[cpu_layer, src_pid].to("cuda", non_blocking=True)
                sk = self.k_swap_scales[cpu_layer, src_pid].to("cuda", non_blocking=True)
                sv = self.v_swap_scales[cpu_layer, src_pid].to("cuda", non_blocking=True)

                self.k_cache[gpu_layer, dst_pid] = dequantize_int8(qk, sk, out_dtype=torch.float16)
                self.v_cache[gpu_layer, dst_pid] = dequantize_int8(qv, sv, out_dtype=torch.float16)

    def _swap_blocks_int8(
        self,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        is_swap_out: bool,
        gpu_layer: int,
        cpu_layer: int,
    ):
        """
        Python fallback for --int8-transfer: quantize on the source side,
        copy INT8 bytes across the PCIe bus, dequantize on the destination
        side. Per-token granularity by default; overridden by
        ``engine_config.quant_granularity``.

        This path is intentionally unfused — ~10x slower than the C++
        swap_blocks kernel on paper — and exists so the int8 wire format
        can be validated end-to-end before a fused CUDA/ISPC kernel lands.

        TODO(gpu-verify): replace with a fused ``swiftllm_c.swap_blocks_int8``
        once that kernel exists. Until then this is correct but slow.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        granularity = QuantGranularity(self.engine_config.quant_granularity)

        src_k, src_v, dst_k, dst_v = (
            (self.k_cache, self.v_cache, self.k_swap, self.v_swap)
            if is_swap_out
            else (self.k_swap, self.v_swap, self.k_cache, self.v_cache)
        )
        src_layer, dst_layer = (gpu_layer, cpu_layer) if is_swap_out else (cpu_layer, gpu_layer)

        for src_pid, dst_pid in zip(src_block_ids, dst_block_ids):
            # Gather src block → quantize → send int8 + scale → dequantize.
            src_k_block = src_k[src_layer, src_pid]
            src_v_block = src_v[src_layer, src_pid]

            qk, sk = quantize_int8(src_k_block, granularity=granularity)
            qv, sv = quantize_int8(src_v_block, granularity=granularity)

            # The actual wire crossing happens on these four .to() calls.
            # Only int8 q and fp32 scale transit the bus (never the
            # original fp16 block) — this is where the bandwidth saving
            # materializes.
            qk = qk.to(dst_k.device, non_blocking=True)
            qv = qv.to(dst_v.device, non_blocking=True)
            sk = sk.to(dst_k.device, non_blocking=True)
            sv = sv.to(dst_v.device, non_blocking=True)

            dst_k[dst_layer, dst_pid] = dequantize_int8(qk, sk, out_dtype=dst_k.dtype)
            dst_v[dst_layer, dst_pid] = dequantize_int8(qv, sv, out_dtype=dst_v.dtype)

    @torch.inference_mode()
    def set_block_tables(
        self,
        mappings: tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]     
    ):
        """
        Establish new mappings in the block tables
        """
        (gpu_vids, gpu_pids), (cpu_vids, cpu_pids) = mappings
        if gpu_vids:
            self.gpu_block_table.view(-1)[gpu_vids] = torch.tensor(gpu_pids, dtype=torch.int32, device="cuda")
        if cpu_vids:
            self.cpu_block_table.view(-1)[cpu_vids] = torch.tensor(cpu_pids, dtype=torch.int32, device="cpu")
