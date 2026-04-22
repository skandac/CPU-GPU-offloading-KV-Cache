"""
benchmarks/run_latency.py

Measure request-level latency:
    - TTFT    : time-to-first-token (arrival → first streamed token)
    - p50/p95/p99 per-token decode latency
    - end-to-end latency distribution

USAGE:
    python benchmarks/run_latency.py \\
        --variant fp16-neo \\
        --workload azure-code \\
        --rate 1.0 \\
        --num-requests 200 \\
        --config evaluation/configs/config-a10-8b.json

OUTPUT:
    benchmarks/results/latency_{variant}_{workload}_r{rate}.csv
        columns: variant, workload, rate_req_s, req_idx,
                 input_len, output_len, ttft_s, e2e_latency_s,
                 per_token_latency_s

    Plus a short summary printed to stdout:
        ttft p50/p95/p99, per-token p50/p95/p99.

CAVEAT on TTFT:
    evaluation/benchmark.py:run_test records only (start, end) timestamps —
    there is no per-chunk trace, so true TTFT requires the server's OpenAI
    streaming API. If --ttft-mode=approx (default), TTFT is estimated as
    end - (output_len * mean_decode_s). For exact TTFT switch to
    --ttft-mode=stream which routes through api_client.py's streaming path.
    (The streaming helper lives in evaluation/api_client.py and is not
    reused here to avoid touching that file.)
    TODO(gpu-verify): implement --ttft-mode=stream once the streaming API
                      on the NEO side is confirmed to match vLLM's.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os

from _common import (
    resolve_variant,
    load_workload,
    ensure_out_dir,
    percentile,
    VARIANTS, WORKLOADS, GRANULARITIES,
    EVAL_DIR,
    start_server, stop_server, run_test,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEO latency harness (M8)")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--quant-granularity", choices=GRANULARITIES, default="per-token",
                   help="INT8 scale granularity (M10; ignored for non-INT8 variants)")
    p.add_argument("--workload", choices=WORKLOADS, default="azure-code")
    p.add_argument("--rate", type=float, default=1.0,
                   help="Offered rate (req/s). Choose below saturation.")
    p.add_argument("--num-requests", type=int, default=200)
    p.add_argument("--config", default=os.path.join(EVAL_DIR, "configs", "config-a10-8b.json"))
    p.add_argument("--ttft-mode", choices=["approx", "stream"], default="approx")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    return p.parse_args()


def _load_records(json_path: str) -> list[dict]:
    with open(json_path) as f:
        return json.load(f)


def _per_request_rows(records: list[dict], args, ttft_mode: str) -> list[dict]:
    rows: list[dict] = []
    # For approx TTFT we need a decode-rate estimate — use the mean observed
    # per-token latency across all records, then subtract that from the e2e.
    decode_s_mean = None
    if ttft_mode == "approx":
        per_toks = [
            (r["end"] - r["start"]) / max(r["output_len"], 1) for r in records
        ]
        if per_toks:
            decode_s_mean = sum(per_toks) / len(per_toks)

    for i, r in enumerate(records):
        e2e = r["end"] - r["start"]
        out_len = max(r["output_len"], 1)
        per_tok = e2e / out_len
        if ttft_mode == "approx":
            # e2e = TTFT + decode_time; decode_time ~= out_len * decode_s_mean
            ttft = max(e2e - out_len * (decode_s_mean or per_tok), 0.0)
        else:  # "stream" — would be populated by a real streaming client
            ttft = float("nan")
        rows.append({
            "variant": args.variant,
            "workload": args.workload,
            "rate_req_s": args.rate,
            "req_idx": i,
            "input_len": r["input_len"],
            "output_len": r["output_len"],
            "ttft_s": ttft,
            "e2e_latency_s": e2e,
            "per_token_latency_s": per_tok,
        })
    return rows


def _print_summary(rows: list[dict]) -> None:
    ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] == r["ttft_s"]]  # skip nan
    pts   = [r["per_token_latency_s"] for r in rows]
    e2es  = [r["e2e_latency_s"] for r in rows]
    print(f"  n = {len(rows)}")
    if ttfts:
        print(f"  TTFT      p50={percentile(ttfts, 50):.3f}s  p95={percentile(ttfts, 95):.3f}s  p99={percentile(ttfts, 99):.3f}s")
    print(f"  per-token p50={percentile(pts, 50):.4f}s  p95={percentile(pts, 95):.4f}s  p99={percentile(pts, 99):.4f}s")
    print(f"  e2e       p50={percentile(e2es, 50):.3f}s  p95={percentile(e2es, 95):.3f}s  p99={percentile(e2es, 99):.3f}s")


async def main_async():
    args = parse_args()
    variant_spec = resolve_variant(args.variant, args.quant_granularity)
    print(f"[info] variant: {args.variant} — {variant_spec.description}")
    g_tag = f"_{args.quant_granularity}" if "int8" in args.variant else ""

    with open(args.config) as f:
        config = json.load(f)

    prompts, output_lens, res_prefix, model_path = load_workload(
        args.workload, variant_spec.server_name, config, args.num_requests
    )
    res_prefix = f"{res_prefix}-{args.variant}"

    start_server(variant_spec.server_name, config)
    try:
        await run_test(prompts, output_lens, res_prefix, model_path, rate=args.rate)
    finally:
        stop_server()

    json_path = f"{res_prefix}-lat-{str(args.rate).replace('.', '_')}.json"
    records = _load_records(json_path)
    rows = _per_request_rows(records, args, args.ttft_mode)

    out_dir = ensure_out_dir(args.out)
    rate_tag = str(args.rate).replace(".", "_")
    csv_path = os.path.join(out_dir, f"latency_{args.variant}{g_tag}_{args.workload}_r{rate_tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {csv_path}")

    print(f"[summary] variant={args.variant} workload={args.workload} rate={args.rate}")
    _print_summary(rows)


if __name__ == "__main__":
    asyncio.run(main_async())
