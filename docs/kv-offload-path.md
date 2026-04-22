# KV-cache offload path

End-to-end tour of how a KV block moves between GPU and CPU in NEO
(swiftllm). The goal here is to make the control-plane ↔ data-plane split
legible: *who decides* a block should move, *who updates the bookkeeping*,
and *who actually copies the bytes*.

This is a read-the-code companion, not an API reference. Everything below
maps to real files and line ranges.

---

## 1. The moving parts

```
                     ┌──────────────────────┐
   request arrives → │  Scheduler           │  (control)
                     │  server/scheduler.py │
                     └──────────┬───────────┘
                                │ decides: swap_in / swap_out / pref_to_cpu
                                ▼
                     ┌──────────────────────────┐
                     │  BlockManager            │  (control)
                     │  server/block_manager.py │
                     └──────────┬───────────────┘
                                │ emits (src_pids, dst_vids, dst_pids)
                                ▼
                     ┌──────────────────────────┐
                     │  kvcache_mgmt.py         │  (dispatch)
                     │  block_swapper.py        │
                     └──────────┬───────────────┘
                                │
                  ┌─────────────┴──────────────┐
                  ▼                            ▼
           GPU block copy                CPU↔GPU copy
           (paged_attn kernel            (csrc/src/
            internal move)                block_swapping.cpp)
                                                │
                                                ▼
                                       ┌────────────────────┐
                                       │  pacpu (CPU attn)  │
                                       │  pacpu/pacpu.ispc  │
                                       └────────────────────┘
                                    (consumes offloaded blocks
                                     during CPU decoding)
```

Two namespaces are in play throughout:

- **VID (virtual block id):** `seq_id * block_table_width + slot_idx`.
  Stable for the lifetime of a request. This is what the model code sees.
- **PID (physical block id):** index into the free pool of the *target
  device*. Reassigned on every alloc/free.

The `block_table` is `vid -> pid`. Every swap is, concretely, an update to
this table plus a byte copy from the source-device PID to the new
dest-device PID.

---

## 2. Scheduler — "should this sequence live on CPU right now?"

[`swiftllm/server/scheduler.py`](../swiftllm/server/scheduler.py) drives
offload policy. `Scheduler` keeps three queues:

- `waiting_q`         — not yet scheduled
- `gpu_decoding_q`    — actively decoding on GPU
- `cpu_decoding_q`    — decoding on CPU (offloaded)

`get_next_batch()` (line 393) picks one of two strategies:

- `_get_next_batch_old` — GPU-only baseline (`always_use_gpu=True`).
- `_get_next_batch_new` — the NEO path. Produces up to two sub-batches
  that can be pipelined with CPU decoding.

The interesting method is `_get_next_batch_new` (line 237). It decides
three sets of requests per iteration:

1. **`swpout_reqs`** — pop victims off the tail of `gpu_decoding_q` until
   the projected GPU block demand fits under `swap_out_threshold`
   (line 253). Victims move onto the front of `cpu_decoding_q`.
2. **`swpin_reqs`** — reverse direction. While the GPU has headroom
   (`swap_in_threshold`, line 256 — 95% of the swap-out level, giving
   hysteresis to prevent thrash), pull CPU-decoding requests back onto
   GPU (line 274).
3. **`pref_to_cpu` / `pref_to_gpu`** — new prefills. A simple heuristic
   (line 300) prefers GPU, falls back to CPU, and once a given batch has
   any CPU prefill every later prefill stays on CPU for fairness.

The assertion `not swpout_reqs or not swpin_reqs` (line 284) enforces
that each iteration is either swapping *in* or *out* — never both. This
is what lets `BlockManager.prepare` reduce the operation to a single
source/destination pair.

`_decide_mode_and_gen_batch` (line 142) picks between sequential and
pipelined execution by comparing throughput (line 225–231). The CPU
queue is only tapped in pipelined mode, so when pipelining loses, CPU
KV blocks just sit on host memory.

---

## 3. BlockManager — "update the table, name the PIDs"

[`swiftllm/server/block_manager.py`](../swiftllm/server/block_manager.py)
is pure bookkeeping — no CUDA, no memcpy.

`DeviceBlockManager` (line 14) holds, per device:

- `seq_num_blks[seq_id]`        — how many blocks this seq currently owns
- `block_table[seq_id, slot]`   — vid → pid mapping
- `is_block_free[split_id]`     — pool bitmap, one per split

The "split" concept (line 34) is the `extra_layer_for_cprf` feature:
CPU-prefill requests first stage into an intermediate split before
landing in the normal CPU pool. Everything else uses split 0.

`_get_new_blk_ids` (line 58) is the allocator: find free bits, flip them,
return PIDs. `alloc` (line 79) translates a list of requests plus
`split_point` into (vids, pids) pairs.

The orchestrator is `prepare` (line 195). Given `batches`, `cur_swap_out`,
`cur_swap_in`, it:

1. `_initiate_swap` (line 172): free blocks on the source device, alloc
   on the destination device, produce `(src_pids, dst_vids, dst_pids)`.
2. `_alloc_blocks_for_batch`: make room for the newly-active step.
3. Handles the cprf split — allocates CPU-intermediate blocks for CPU
   prefills and stashes their PIDs on the `SubBatch` as `src_blk_ids` /
   `dst_blk_ids` (line 256), which the data plane consumes later.

Return value is the tuple `(mappings, swappings, is_swap_out)` that the
worker uses to (a) patch its own GPU-resident block table and (b)
enqueue the actual copy.

`update_and_free` (line 264) runs at iteration end: advances outputs,
frees blocks for finished sequences on both devices.

---

## 4. Worker dispatch — `kvcache_mgmt` / `block_swapper`

These two files translate the control-plane tuple into kernel launches.
`kvcache_mgmt.py` owns the *block table on the GPU* (the device-resident
copy of `BlockManager.block_table`) and applies the `(dst_vids, dst_pids)`
update as a scatter. `block_swapper.py` stages the actual H2D / D2H.

The pattern for a swap-out iteration:

1. Control plane calls `BlockManager.prepare(...)` — new CPU PIDs come
   out.
2. Worker updates its CPU-side `block_table` on-device (GPU tensor).
3. Worker posts copy: for each `(src_pid_gpu, dst_pid_cpu)`, call into
   the csrc swapping kernel.
4. Worker launches the normal forward pass. Note the copy is overlapped
   with compute on a separate CUDA stream — that's what makes the
   pipelined mode profitable.

---

## 5. Data plane — `csrc/src/block_swapping.cpp`

This is the low-level mover. Two entry points land here via the
pybind layer in `csrc/src/entrypoints.cpp`:

- `swap_blocks(src, dst, src_pids, dst_pids)` — element-wise gather of
  whole KV blocks from one tensor into another, launched as a CUDA
  kernel when src/dst are both GPU-resident, or as a pinned-buffer
  memcpy when one side is host.
- block-granularity copies for the cprf intermediate path.

The CPU side of the KV cache is a pinned host tensor allocated at
engine init so every copy is a DMA, not a pageable `memcpy`. That's the
only reason the offload path is latency-tolerable.

---

## 6. CPU attention — `pacpu`

Once a block is on the CPU, `cpu_decoding_q` requests need attention
computed against it. That work runs in
[`pacpu/pacpu.ispc`](../pacpu/pacpu.ispc) (ISPC-vectorized; AVX2/AVX512
fan-out), glued in via
[`pacpu/pacpu.cpp`](../pacpu/pacpu.cpp). The header
[`pacpu/core.h`](../pacpu/core.h) defines the block layout the ISPC code
expects — critically the same block size as the GPU side so the copy
is a straight memcpy, no reshuffle.

The scheduler's "pipelined mode" check (scheduler.py:225) is comparing
(GPU-only sequential time) vs (GPU batch + CPU batch overlapped). The
CPU side of that overlap is pacpu. If pacpu is too slow for the
current block count, `_decide_mode_and_gen_batch` falls back to
`gpu_only_batch` (line 234) — CPU KV blocks stay resident but are not
touched this iteration.

---

## 7. Failure modes worth knowing

- **`No enough free blocks available on ...`** (block_manager.py:67) —
  the scheduler over-committed the device. Means the swap-out threshold
  logic produced an underestimate, or a prefill pushed past
  `num_gpu_blocks`. Reproduces most often when `max_tokens_in_batch`
  and `num_gpu_blocks` are configured inconsistently.
- **Swap thrash** — if `swap_in_threshold` and `swap_out_threshold` were
  equal, a single decode step could oscillate. The 0.95 factor
  (scheduler.py:256) is the anti-thrash gap.
- **`Cannot swap in to intermediate space`** (block_manager.py:183) —
  caller tried `is_swap_out=False, use_itm=True`. The intermediate
  split only exists as a staging area for CPU prefills going host-ward.

---

## 8. Reading order for new contributors

1. `server/scheduler.py` — policy, top-down.
2. `server/block_manager.py` — just `prepare()` and `_initiate_swap()`.
3. `worker/kvcache_mgmt.py` + `worker/block_swapper.py` — the dispatch
   glue.
4. `csrc/src/block_swapping.cpp` + `csrc/src/entrypoints.cpp` — the
   actual copy.
5. `pacpu/pacpu.ispc` — only if you're touching CPU attention.

Skip `_get_next_batch_old` on a first pass; it's the baseline path
retained for `always_use_gpu` and doesn't exercise the offload logic.
