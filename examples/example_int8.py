"""
INT8 KV-cache transfer demo.

Exercises the same offline inference path as ``example.py`` but with the
INT8 quantization flags enabled. This is the smoke test referenced from
the runbook in ``ec2_run_instructions.md`` §7.2 — if it generates
coherent text, the INT8 path is wired up end to end.

Two flags are added on top of the base example:

  --int8-transfer        Enable the INT8 H2D/D2H transfer path (M7).
  --int8-cpu-kv          Store CPU-resident KV blocks as INT8 (M6).
  --quant-granularity    {per-token, per-channel, per-token-per-head}
                         Default: per-token. See docs/int8-design.md §3.

Per-token granularity is the recommended default and matches the layout
used by Q8 attention work in the literature. The ``--int8-transfer``
flag is composable with ``--int8-cpu-kv`` — together they keep CPU-side
blocks in INT8 throughout the swap path with no FP16 staging.

USAGE
-----
    python examples/example_int8.py \\
        --model-path /path/to/Llama-3-8B \\
        --model-name llama3_8b \\
        --int8-transfer --quant-granularity per-token

To run with both INT8 storage AND INT8 transfer::

    python examples/example_int8.py \\
        --model-path /path/to/Llama-3-8B \\
        --model-name llama3_8b \\
        --int8-cpu-kv --int8-transfer \\
        --quant-granularity per-token

NOTES
-----
- The FP16 baseline behaviour is exactly ``examples/example.py`` with no
  quantization flags. Use that script to A/B against this one.
- Round-trip correctness of the quantizer is exercised by
  ``tests/test_int8_transfer.py`` (CPU-only, no model required).
- For perplexity and latency comparisons, see ``benchmarks/`` and
  ``scripts/run_ablation.sh``.
"""
from __future__ import annotations

import argparse
import os
import time

from transformers import AutoTokenizer

import swiftllm


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    repo_dir = os.path.dirname(script_dir)
    parser = argparse.ArgumentParser(
        description="Offline INT8 inference demo for the NEO engine.",
    )
    parser.add_argument("--model-path", required=True,
                        help="Path to the model weights directory.")
    parser.add_argument("--model-name", required=True,
                        help="Lowercase model identifier (e.g. llama3_8b).")
    parser.add_argument("--tp-degree", type=int, default=1)
    parser.add_argument("--profile-result-path", type=str,
                        default=f"{repo_dir}/profile_results/")
    parser.add_argument("--num-gpu-blocks", type=int, default=50)
    parser.add_argument("--swap-space", type=int, default=4,
                        help="GB of pinned host memory for CPU-resident KV.")
    parser.add_argument("--prompt-path", type=str,
                        default=f"{script_dir}/example.txt")

    # ---- INT8-specific flags (M6/M7/M10) ----
    parser.add_argument("--int8-transfer", action="store_true",
                        help="INT8 H2D/D2H transfer path (M7).")
    parser.add_argument("--int8-cpu-kv", action="store_true",
                        help="Store CPU-resident KV blocks as INT8 (M6).")
    parser.add_argument(
        "--quant-granularity",
        choices=["per-token", "per-channel", "per-token-per-head"],
        default="per-token",
        help="INT8 scale granularity (default: per-token).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[demo] model={args.model_name} path={args.model_path}")
    print(f"[demo] int8_transfer={args.int8_transfer} "
          f"int8_cpu_kv={args.int8_cpu_kv} "
          f"granularity={args.quant_granularity}")

    engine_config = swiftllm.EngineConfig(
        model_path=args.model_path,
        model_name=args.model_name,
        tensor_parallel_degree=args.tp_degree,
        profile_result_path=args.profile_result_path,
        num_gpu_blocks_override=args.num_gpu_blocks,
        swap_space=args.swap_space,
        # INT8 toggles (engine-level — see swiftllm/engine_config.py):
        int8_transfer=args.int8_transfer,
        int8_cpu_kv=args.int8_cpu_kv,
        quant_granularity=args.quant_granularity,
    )

    t0 = time.time()
    model = swiftllm.LlamaModel(engine_config)
    print(f"[demo] engine ready in {time.time() - t0:.2f}s")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    with open(args.prompt_path) as f:
        prompt = f.read().strip()

    input_ids = tokenizer.encode(prompt, return_tensors=None)
    print(f"[demo] prompt tokens: {len(input_ids)}")

    request = swiftllm.create_request(input_ids, max_tokens=64)
    output_ids = model.generate([request])[0]
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
