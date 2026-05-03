# Examples

Offline inference demos for the NEO engine. Each script loads the engine,
runs a single prompt through it, and prints the generation. Useful for
sanity-checking that the install + weight download + pacpu build all
succeeded before spending GPU time on benchmarks.

## Files

| Script | What it demonstrates |
|---|---|
| `example.py` | Baseline FP16 inference — the original NEO offline example. Run this first to confirm the engine works. |
| `example_int8.py` | The same flow with the INT8 KV-cache transfer / CPU-KV flags wired in. Use this to smoke-test the INT8 path before the full benchmark sweeps. |
| `example.txt` | A multi-paragraph English prompt used by both scripts. |

## Quick start

```bash
# FP16 baseline
python examples/example.py \
    --model-path /path/to/Llama-3-8B \
    --model-name llama3_8b

# INT8 transfer (M7) — INT8 on the wire, FP16 on device
python examples/example_int8.py \
    --model-path /path/to/Llama-3-8B \
    --model-name llama3_8b \
    --int8-transfer --quant-granularity per-token

# INT8 transfer + INT8 CPU storage (M6 + M7)
python examples/example_int8.py \
    --model-path /path/to/Llama-3-8B \
    --model-name llama3_8b \
    --int8-cpu-kv --int8-transfer --quant-granularity per-token
```

If `example_int8.py` produces coherent generation, the INT8 path is
wired up end to end. Round-trip numerical correctness of the quantizer
is exercised separately by `tests/test_int8_transfer.py` (CPU-only).

For full performance and quality measurements, see:

- `benchmarks/run_latency.py` — load-latency curves
- `benchmarks/run_throughput.py` — throughput sweeps
- `benchmarks/run_perplexity.py` — WikiText-2 perplexity
- `scripts/run_ablation.sh` — granularity ablation (M10)
