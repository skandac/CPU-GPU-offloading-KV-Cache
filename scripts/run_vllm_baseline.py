"""
scripts/run_vllm_baseline.py

Runs vLLM (and optionally NEO) on two workloads — OpenAI Summarization (osc)
and Azure Code — producing CSV result files for head-to-head comparison.

This script REUSES the existing evaluation/server.py and evaluation/benchmark.py
machinery to launch vLLM in the exact same configuration as the paper's
reproduction scripts. It then post-processes the JSON outputs into CSVs.

USAGE (from repo root, with conda env 'neo' active):
    python scripts/run_vllm_baseline.py [flags]

FLAGS:
    --system {vllm,ours,base}   Which server to launch (default: vllm)
                                 vllm → baseline
                                 ours → NEO (for generating NEO CSVs too)
                                 base → NEO with --always-use-gpu
    --config PATH                Path to eval config JSON
                                 (default: evaluation/configs/config-t4-7b.json)
    --workloads osc[,azure-code] Comma-separated list of workloads to run
                                 (default: "osc,azure-code")
    --rates 0.2,0.4,...          Comma-separated request rates (req/s).
                                 Defaults: vllm→[0.2,0.4,0.5,0.6],
                                           ours→[0.5,1.5,2.5,3.1,3.5,3.7,3.9]
    --num-requests N             Number of requests per run (default: 100)
    --out-dir PATH               CSV output dir (default: evaluation/csv_results/)

OUTPUTS:
    evaluation/csv_results/{system}_{workload}_requests.csv  (per-request CSV)
    evaluation/csv_results/{system}_{workload}_summary.csv   (aggregate CSV)
    evaluation/results/*.json                                (raw JSONs, reused)

CSV SCHEMAS:
    {system}_{workload}_requests.csv columns:
        system, workload, rate_req_s, req_idx, input_len, output_len,
        start_s, end_s, latency_s, per_token_latency_s
    {system}_{workload}_summary.csv columns:
        system, workload, rate_req_s, n_requests,
        throughput_req_s, avg_per_token_latency_s,
        p50_latency_s, p95_latency_s, p99_latency_s

DO NOT RUN ON THE M4 MAC — requires CUDA + the NEO install.
"""

import argparse
import asyncio
import csv
import json
import os
import random
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Make evaluation/ importable — the reproduce scripts do the same by cd-ing
# into evaluation/ before running, but we want to run from repo root.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
EVAL_DIR = os.path.join(REPO_DIR, "evaluation")
sys.path.insert(0, EVAL_DIR)

# pylint: disable=wrong-import-position,import-error
from server import start_server, stop_server          # noqa: E402
from benchmark import run_test, prepare_real_test     # noqa: E402


# =============================================================================
# Workload preparation
# =============================================================================

def _get_rand_array_seeded(n: int, avg_val: int, ratio: float, seed: int) -> list:
    """
    Deterministic variant of benchmark.py:_get_rand_array (seeds the RNG).
    Used for synthetic Azure Code workload.
    """
    rng = random.Random(seed)
    delta = int(avg_val * ratio)
    return [avg_val + rng.randint(-delta, delta) for _ in range(n)]


def prepare_azure_code_workload(
    nreqs: int,
    server_name: str,
    config: dict,
) -> tuple:
    """
    Prepare an Azure Code workload. Prefers a real dataset file if it exists
    at evaluation/data/azure-code-{MODEL}.json (same schema as osc dataset:
    list of {"prompt": int, "max_tokens": int}). Falls back to a synthetic
    workload with distributional characteristics approximating the Azure
    LLMInferenceTrace 2023 dataset (Splitwise paper, Patel et al. ISCA'24):
        - input_len ~ log-normal, median ~250, heavy tail to ~1000
        - output_len ~ log-normal, median ~80, heavy tail to ~400
    TODO(gpu-verify): if exact Azure Code traces are available, replace the
                      synthetic fallback by placing the JSON in evaluation/data/.

    Returns: (prompts, output_lens, res_file_prefix, model_path)
             — same shape as benchmark.py:prepare_real_test/prepare_mock_test.
    """
    data_path = os.path.join(EVAL_DIR, "data", f"azure-code-{config['model']}.json")

    if os.path.exists(data_path):
        # Real dataset: mirror prepare_real_test's logic exactly
        with open(data_path) as f:
            datas = json.load(f)[:nreqs]
        prompts = [[10] * d["prompt"] for d in datas]
        output_lens = [d["max_tokens"] for d in datas]
    else:
        # Synthetic fallback — deterministic via fixed seeds so re-runs are
        # comparable across systems.
        print(f"[warn] {data_path} missing — using synthetic Azure Code workload.")
        print(f"[warn] For exact traces, drop a real JSON file at that path.")
        # TODO(gpu-verify): double-check these distribution parameters against
        # the Splitwise paper tables (Patel et al., ISCA '24, Fig 4).
        input_lens = _get_rand_array_seeded(nreqs, avg_val=250, ratio=0.4, seed=0xA2C0)
        output_lens = _get_rand_array_seeded(nreqs, avg_val=80, ratio=0.4, seed=0xA2C1)
        prompts = [[10] * il for il in input_lens]

    # Match the res_file naming convention used by benchmark.py
    # (this is the PREFIX — benchmark.py appends "-lat-{rate}.json" at run_test time)
    cur_dir = os.path.join(EVAL_DIR, "results")
    os.makedirs(cur_dir, exist_ok=True)
    res_file_prefix = f"{cur_dir}/{server_name}-azure-code"

    return prompts, output_lens, res_file_prefix, config["model_path"]


# =============================================================================
# CSV conversion — runs AFTER all benchmark JSONs are written
# =============================================================================

def _per_token_latency(start: float, end: float, output_len: int) -> float:
    """Mirror illustrator.py:get_lat_avg's per-request metric."""
    return (end - start) / output_len if output_len > 0 else float("nan")


def _percentile(values: list, p: float) -> float:
    """Simple percentile (no numpy dependency required here — numpy is fine,
       but keeping this self-contained for clarity)."""
    if not values:
        return float("nan")
    vs = sorted(values)
    k = (len(vs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vs) - 1)
    if f == c:
        return vs[f]
    return vs[f] + (vs[c] - vs[f]) * (k - f)


def write_csvs(
    system: str,
    workload: str,
    rates: list,
    res_file_prefix: str,
    out_dir: str,
) -> None:
    """
    For each rate, read the JSON file written by benchmark.py:run_test and
    append rows to per-request and summary CSVs.

    Expected JSON filename:
        {res_file_prefix}-lat-{rate_with_underscore}.json
    """
    os.makedirs(out_dir, exist_ok=True)
    req_csv_path = os.path.join(out_dir, f"{system}_{workload}_requests.csv")
    sum_csv_path = os.path.join(out_dir, f"{system}_{workload}_summary.csv")

    req_headers = [
        "system", "workload", "rate_req_s", "req_idx",
        "input_len", "output_len", "start_s", "end_s",
        "latency_s", "per_token_latency_s",
    ]
    sum_headers = [
        "system", "workload", "rate_req_s", "n_requests",
        "throughput_req_s", "avg_per_token_latency_s",
        "p50_latency_s", "p95_latency_s", "p99_latency_s",
    ]

    with open(req_csv_path, "w", newline="") as req_f, \
         open(sum_csv_path, "w", newline="") as sum_f:
        req_writer = csv.writer(req_f)
        sum_writer = csv.writer(sum_f)
        req_writer.writerow(req_headers)
        sum_writer.writerow(sum_headers)

        for rate in rates:
            rate_str = str(rate).replace(".", "_")
            json_path = f"{res_file_prefix}-lat-{rate_str}.json"

            if not os.path.exists(json_path):
                print(f"[warn] missing result file: {json_path} (skipping rate={rate})")
                continue

            with open(json_path) as jf:
                records = json.load(jf)

            latencies = []
            per_tok_lats = []
            for i, rec in enumerate(records):
                lat = rec["end"] - rec["start"]
                ptl = _per_token_latency(rec["start"], rec["end"], rec["output_len"])
                latencies.append(lat)
                per_tok_lats.append(ptl)
                req_writer.writerow([
                    system, workload, rate, i,
                    rec["input_len"], rec["output_len"],
                    f"{rec['start']:.6f}", f"{rec['end']:.6f}",
                    f"{lat:.6f}", f"{ptl:.6f}",
                ])

            n = len(records)
            # Throughput: match illustrator.py's trim semantics for req/s:
            #   sorted completion times, drop first 10%, last 30%, 1/mean(gaps)
            # Using the same math so numbers are comparable to fig10a metrics.
            ends = sorted(r["end"] for r in records)
            if n >= 2:
                lo = n // 10
                hi = n - n // 10 * 3 + 1
                window = ends[lo:hi]
                if len(window) >= 2:
                    throughput = (len(window) - 1) / (window[-1] - window[0])
                else:
                    throughput = float("nan")
            else:
                throughput = float("nan")

            avg_ptl = sum(per_tok_lats) / n if n else float("nan")
            p50 = _percentile(latencies, 50)
            p95 = _percentile(latencies, 95)
            p99 = _percentile(latencies, 99)

            sum_writer.writerow([
                system, workload, rate, n,
                f"{throughput:.6f}", f"{avg_ptl:.6f}",
                f"{p50:.6f}", f"{p95:.6f}", f"{p99:.6f}",
            ])

    print(f"[ok] wrote {req_csv_path}")
    print(f"[ok] wrote {sum_csv_path}")


# =============================================================================
# One-round driver (start server → run across rates → stop server)
# =============================================================================

async def run_one_round(
    system: str,
    workloads: list,
    rates: list,
    num_requests: int,
    config: dict,
    out_dir: str,
) -> None:
    """
    Start the chosen server, iterate over workloads × rates, then stop the
    server. After the server is down, write CSVs from the JSON outputs.

    NOTE: start_server/stop_server are imported from evaluation/server.py
    and have the exact semantics used by reproduce-fig6c.py / fig10a.py.
    """
    start_server(system, config)
    try:
        for workload in workloads:
            if workload == "osc":
                # prepare_real_test handles osc data loading + capping to 100 reqs
                prompts, output_lens, res_prefix, model_path = prepare_real_test(
                    "osc", config, system,
                )
                # prepare_real_test caps to 100 internally. Respect --num-requests.
                prompts = prompts[:num_requests]
                output_lens = output_lens[:num_requests]
            elif workload == "azure-code":
                prompts, output_lens, res_prefix, model_path = prepare_azure_code_workload(
                    num_requests, system, config,
                )
            else:
                print(f"[warn] unknown workload: {workload} (skipping)")
                continue

            print(f"[info] {system} × {workload}: {len(prompts)} requests "
                  f"across rates {rates}")
            for rate in rates:
                await run_test(prompts, output_lens, res_prefix, model_path, rate=rate)

            # After all rates for this workload are done, flush CSVs
            write_csvs(system, workload, rates, res_prefix, out_dir)
    finally:
        stop_server()
    await asyncio.sleep(5)   # match the cooldown in the reproduce scripts


# =============================================================================
# Main
# =============================================================================

DEFAULT_RATES = {
    "vllm": [0.2, 0.4, 0.5, 0.6],
    "ours": [0.5, 1.5, 2.5, 3.1, 3.5, 3.7, 3.9],
    "base": [0.5, 1.0, 1.5, 2.0],   # rough baseline rates — TODO(gpu-verify)
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--system", choices=["vllm", "ours", "base"], default="vllm")
    p.add_argument(
        "--config",
        default=os.path.join(EVAL_DIR, "configs", "config-t4-7b.json"),
        help="Path to evaluation config JSON",
    )
    p.add_argument("--workloads", default="osc,azure-code",
                   help="Comma-separated workloads (osc, azure-code)")
    p.add_argument("--rates", default=None,
                   help="Comma-separated rates (req/s). Defaults depend on --system.")
    p.add_argument("--num-requests", type=int, default=100,
                   help="Max requests per workload (default: 100, paper: 2000)")
    p.add_argument("--out-dir",
                   default=os.path.join(EVAL_DIR, "csv_results"),
                   help="Directory for CSV outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load the eval config (same fields used by evaluation/server.py)
    with open(args.config) as f:
        config = json.load(f)

    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    if args.rates:
        rates = [float(r.strip()) for r in args.rates.split(",") if r.strip()]
    else:
        rates = DEFAULT_RATES[args.system]

    print(f"[info] system={args.system} config={args.config}")
    print(f"[info] workloads={workloads} rates={rates}")
    print(f"[info] num_requests={args.num_requests} out_dir={args.out_dir}")

    # Run from EVAL_DIR so any relative paths (profile_results/, *.log) resolve
    # the same way they do under reproduce-fig6c.py.
    orig_cwd = os.getcwd()
    try:
        os.chdir(EVAL_DIR)
        asyncio.run(run_one_round(
            system=args.system,
            workloads=workloads,
            rates=rates,
            num_requests=args.num_requests,
            config=config,
            out_dir=args.out_dir,
        ))
    finally:
        os.chdir(orig_cwd)

    print("[done] all CSVs written to:", args.out_dir)


if __name__ == "__main__":
    main()
