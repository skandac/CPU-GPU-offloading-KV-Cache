"""
benchmarks/run_throughput.py

Measure steady-state throughput (req/s actually served) across one or more
workloads. Sweeps a list of *offered* rates; records effective throughput.
When a system saturates, effective < offered — that gap is the interesting
signal for head-to-head plots.

USAGE (from repo root, with 'neo' conda env active):
    python benchmarks/run_throughput.py \\
        --variant int8-transfer \\
        --workloads azure-code,osc \\
        --rates 0.5,1.0,2.0,3.0 \\
        --num-requests 200 \\
        --config evaluation/configs/config-a10-8b.json \\
        --out benchmarks/results/

OUTPUTS:
    benchmarks/results/throughput_{variant}_{workload}.csv
        columns: variant, workload, offered_rate_req_s, nreqs,
                 effective_throughput_req_s, duration_s

DO NOT RUN ON A LAPTOP — requires CUDA + NEO install. CPU-only smoke test
would require stubbing start_server, which this script does not do.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys

from _common import (
    resolve_variant,
    load_workload,
    ensure_out_dir,
    VARIANTS, WORKLOADS, GRANULARITIES,
    EVAL_DIR,
    start_server, stop_server, run_test,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEO throughput harness (M8)")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--quant-granularity", choices=GRANULARITIES, default="per-token",
                   help="INT8 scale granularity (M10 ablation; ignored for non-INT8 variants)")
    p.add_argument("--workloads", default="azure-code,osc,synthetic",
                   help="Comma-separated list from: " + ",".join(WORKLOADS))
    p.add_argument("--rates", default="0.5,1.0,2.0,3.0",
                   help="Comma-separated offered rates in req/s")
    p.add_argument("--num-requests", type=int, default=200)
    p.add_argument("--config", default=os.path.join(EVAL_DIR, "configs", "config-a10-8b.json"))
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    return p.parse_args()


def _effective_throughput(json_path: str) -> tuple[float, float, int]:
    """
    Read the JSON written by benchmark.run_test and compute (throughput,
    duration, nreqs). Uses the same trim-10%-start / 30%-tail windowing as
    benchmark.py to avoid warm-up / cool-down contamination.
    """
    with open(json_path) as f:
        records = json.load(f)
    if not records:
        return float("nan"), 0.0, 0
    ends = sorted(r["end"] for r in records)
    n = len(ends)
    window = ends[n // 10: n - n // 10 * 3 + 1]
    if len(window) < 2:
        return float("nan"), 0.0, n
    duration = window[-1] - window[0]
    thrpt = (len(window) - 1) / duration if duration > 0 else float("nan")
    return thrpt, duration, n


async def _run_variant(args, variant_spec, workload: str, config: dict) -> list[dict]:
    rates = [float(x) for x in args.rates.split(",")]
    prompts, output_lens, res_prefix, model_path = load_workload(
        workload, variant_spec.server_name, config, args.num_requests
    )
    # Overlay variant-specific res_prefix suffix so different variants don't
    # collide on cached JSON files.
    res_prefix = f"{res_prefix}-{args.variant}"

    rows = []
    # TODO(gpu-verify): start_server doesn't currently accept extra args;
    # M6/M7 flags should be plumbed through evaluation/server.py. Until
    # then, this harness relies on the relevant flags being hard-coded or
    # set via env vars inside server.py.
    start_server(variant_spec.server_name, config)
    try:
        for rate in rates:
            await run_test(prompts, output_lens, res_prefix, model_path, rate=rate)
            json_path = f"{res_prefix}-lat-{str(rate).replace('.', '_')}.json"
            thrpt, dur, n = _effective_throughput(json_path)
            rows.append({
                "variant": args.variant,
                "workload": workload,
                "offered_rate_req_s": rate,
                "nreqs": n,
                "effective_throughput_req_s": thrpt,
                "duration_s": dur,
            })
    finally:
        stop_server()
    return rows


def _write_csv(out_dir: str, variant: str, workload: str, rows: list[dict]) -> str:
    path = os.path.join(out_dir, f"throughput_{variant}_{workload}.csv")
    fields = ["variant", "workload", "offered_rate_req_s", "nreqs",
              "effective_throughput_req_s", "duration_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {path}")
    return path


async def main_async():
    args = parse_args()
    variant_spec = resolve_variant(args.variant, args.quant_granularity)
    print(f"[info] variant: {args.variant} — {variant_spec.description}")

    with open(args.config) as f:
        config = json.load(f)

    out_dir = ensure_out_dir(args.out)

    # Tag the output filename with granularity so M10 ablation runs don't
    # collide. Non-INT8 variants always use per-token (no effect).
    g_tag = f"_{args.quant_granularity}" if "int8" in args.variant else ""

    for workload in [w.strip() for w in args.workloads.split(",") if w.strip()]:
        print(f"[info] running workload={workload}")
        rows = await _run_variant(args, variant_spec, workload, config)
        _write_csv(out_dir, f"{args.variant}{g_tag}", workload, rows)


if __name__ == "__main__":
    asyncio.run(main_async())
