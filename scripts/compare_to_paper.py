"""
scripts/compare_to_paper.py

Loads the raw JSON result files produced by evaluation/reproduce-fig6c.py and
evaluation/reproduce-fig10a.py, recomputes the same metrics used by
evaluation/illustrator.py, and prints a comparison table showing % deviation
from the numbers reported in the NEO MLSys'25 paper
(https://arxiv.org/abs/2411.01142)..

USAGE (from repo root, with the 'neo' conda env active):
    python scripts/compare_to_paper.py [--results-dir PATH]

FLAGS:
    --results-dir   Path to evaluation/results/ directory.
                    Default: <repo_root>/evaluation/results/

OUTPUT:
    A formatted table for each figure showing:
      - the configuration point (rate or output_len)
      - the reproduced value
      - the paper value (hardcoded below)
      - % deviation = (reproduced - paper) / paper * 100

IMPORTANT NOTES ON EXPECTED DEVIATIONS:
    Fig 6c (load-latency):
        The reproduce script uses 100 requests by default (paper uses 2000).
        Fewer requests → less queuing → lower latency at all rates.
        Expect reproduced latency to be 10–50% LOWER than paper numbers.
        This is expected and documented in evaluation/reproduce-fig6c.py.
        A negative deviation here is NOT a bug.

    Fig 10a (throughput sensitivity):
        The reproduce script uses 2000 requests (same as paper) but only draws
        the x16large line. Throughput ratios should be within ~5% of paper if
        the machine matches (g5.16xlarge / A10G, 32 CPU cores, 256 GB RAM).
        Larger deviations indicate a hardware mismatch or config difference.

TODO(gpu-verify): After running on GPU, update PAPER_NUMBERS below by reading
                  values directly from the paper PDF figures.
                  Current values are best estimates from:
                    - README.md explicit statements
                    - Figure axis limits and described behavior
                  Mark any confirmed values with # CONFIRMED.
"""

import argparse
import json
import os
import sys

import numpy as np


# =============================================================================
# HARDCODED PAPER NUMBERS
# =============================================================================
# Source: NEO MLSys'25 paper, Figure 6c and Figure 10a.
# Hardware: g4dn.4xlarge (T4, 8 CPU cores) for 6c;
#           g5.16xlarge  (A10G, 32 CPU cores) for 10a.
#
# HOW TO READ THESE FROM THE PDF:
#   1. Open the paper PDF (link in README).
#   2. For Fig 6c: hover over each data point on the "VLLM" and "Ours" curves;
#      the y-axis is "Average per token latency (s)".
#   3. For Fig 10a: read the "x16large" curve y-values at each x tick
#      (output lengths 50, 100, 200, 300, 400); y-axis is "Relative throughput"
#      (ours / base). baseline = red dashed line at y=1.0.
#
# TODO(gpu-verify): Replace all ESTIMATED values with values read from the PDF.
# =============================================================================

# ---- Figure 6c: average per-token latency (seconds) ----
# Metric: mean of (end-start)/output_len over the latter 75% of requests
#         (see evaluation/illustrator.py:get_lat_avg)
#
# vLLM baseline at request rates [0.2, 0.4, 0.5, 0.6] req/s
# At low load latency is near the minimum model forward-pass time.
# At rate=0.6 vLLM approaches saturation, latency climbs sharply.
#
# TODO(gpu-verify): all values below marked ESTIMATED — read from PDF Fig 6c.
PAPER_FIG6C = {
    # "vllm": { rate (req/s) -> avg_per_token_latency (s) }
    "vllm": {
        0.2: 0.055,   # ESTIMATED — low-load baseline, T4+Llama2-7B
        0.4: 0.075,   # ESTIMATED — moderate load
        0.5: 0.130,   # ESTIMATED — approaching saturation
        0.6: 0.700,   # ESTIMATED — near saturation, latency climbs sharply
    },
    # "ours" (NEO) at request rates [0.5, 1.5, 2.5, 3.1, 3.5, 3.7, 3.9] req/s
    # NEO offloads KV cache to CPU, sustaining higher rates before saturating.
    "ours": {
        0.5: 0.055,   # ESTIMATED — same low load, similar to vLLM
        1.5: 0.065,   # ESTIMATED — NEO handles higher rate smoothly
        2.5: 0.080,   # ESTIMATED
        3.1: 0.110,   # ESTIMATED
        3.5: 0.180,   # ESTIMATED
        3.7: 0.350,   # ESTIMATED
        3.9: 1.200,   # ESTIMATED — approaching NEO saturation
    },
}

# ---- Figure 10a: relative throughput (ours / base) ----
# Metric: computed by get_tp() in illustrator.py — measures req/s from
#         sorted completion times, trimming outer 30% for warmup/cooldown.
# x-axis: output_lens = [50, 100, 200, 300, 400]
# y-axis: ours_throughput / base_throughput  (1.0 = no gain)
#
# From the README: "NEO achieves up to 12.2%, 13.3%, 29.7%, and 79.3% higher
# throughput over the baseline under different CPU capacities."
# These 4 percentages correspond to g5.{2,4,8,16}xlarge instances.
# The reproduce script draws only the x16large line → max gain ≈ 79.3%.
#
# At output_len=400 the gain is 79.3% (from README, CONFIRMED).
# At shorter output_lens, the KV cache is smaller → less benefit from offload.
#
# TODO(gpu-verify): values for output_lens 50–300 are ESTIMATED.
#                   Read the x16large curve from PDF Fig 10a.
PAPER_FIG10A = {
    # output_len -> relative_throughput (ours / base)
    50:  1.030,   # ESTIMATED — short outputs: mostly compute-bound, little gain
    100: 1.060,   # ESTIMATED
    200: 1.150,   # ESTIMATED
    300: 1.400,   # ESTIMATED
    400: 1.793,   # CONFIRMED (README: "79.3% higher throughput" for x16large)
}


# =============================================================================
# Metric computation — mirrors evaluation/illustrator.py exactly
# =============================================================================

def get_lat_avg(filepath: str) -> float:
    """
    Average per-token latency for a latency-test result file.
    Mirrors illustrator.py:get_lat_avg():
        - Skip first 25% of requests (warm-up)
        - avg of (end-start) / output_len for remaining requests
    """
    with open(filepath) as f:
        data = json.load(f)
    # Drop the first quarter (warm-up effect)
    data = data[len(data) // 4:]
    if not data:
        raise ValueError(f"No data after trimming in {filepath}")
    return sum((x["end"] - x["start"]) / x["output_len"] for x in data) / len(data)


def get_tp(filepath_base: str, filepath_ours: str, interv: tuple) -> float:
    """
    Relative throughput (ours / base) for a pair of throughput-test result files.
    Mirrors illustrator.py:get_tp():
        - Sort completion times
        - Trim outer (interv[0]*100)% and (1-interv[1])*100)% for warmup/cooldown
        - throughput = 1 / mean(inter-completion gaps) in trimmed window
    """
    def _tp_from_file(path: str) -> float:
        with open(path) as f:
            data = json.load(f)
        times = sorted(d["end"] for d in data)
        gaps = [times[j] - times[j - 1] for j in range(1, len(times))]
        n = len(gaps)
        lo = round((n + 1) * interv[0])
        hi = round((n + 1) * interv[1])
        trimmed = gaps[lo:hi]
        if not trimmed:
            raise ValueError(f"No gaps after trimming in {path}")
        return 1.0 / float(np.mean(trimmed))

    tp_base = _tp_from_file(filepath_base)
    tp_ours = _tp_from_file(filepath_ours)
    return tp_ours / tp_base


# =============================================================================
# Table printing
# =============================================================================

def pct_dev(reproduced: float, paper: float) -> float:
    """% deviation: (reproduced - paper) / paper * 100"""
    if paper == 0:
        return float("nan")
    return (reproduced - paper) / paper * 100.0


def print_table(rows: list, headers: list):
    """Print a simple fixed-width table."""
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


# =============================================================================
# Main comparison logic
# =============================================================================

def compare_fig6c(results_dir: str):
    """
    Compare reproduced fig6c latency numbers against paper values.
    Result file pattern: {results_dir}/{sys}-osc-lat-{rate_str}.json
    """
    print("\n" + "=" * 70)
    print("FIGURE 6c — Load-Latency (avg per-token latency, seconds)")
    print("Dataset: OpenAI summarization (osc), T4 + Llama-2-7B")
    print("NOTE: reproduced run uses 100 requests; paper uses 2000.")
    print("      Expect reproduced latency ~10–50% LOWER than paper.")
    print("=" * 70)

    any_found = False
    rows = []

    for sys_name, rate_map in PAPER_FIG6C.items():
        for rate, paper_val in sorted(rate_map.items()):
            rate_str = str(rate).replace(".", "_")
            # Match the filename produced by benchmark.py:run_test
            fname = os.path.join(results_dir, f"{sys_name}-osc-lat-{rate_str}.json")

            if not os.path.exists(fname):
                rows.append((sys_name, f"{rate:.1f}", "—  (file missing)", f"{paper_val:.4f}", "—"))
                continue

            any_found = True
            try:
                reproduced = get_lat_avg(fname)
                dev = pct_dev(reproduced, paper_val)
                rows.append((
                    sys_name,
                    f"{rate:.1f}",
                    f"{reproduced:.4f}",
                    f"{paper_val:.4f}",
                    f"{dev:+.1f}%",
                ))
            except Exception as exc:
                rows.append((sys_name, f"{rate:.1f}", f"ERROR: {exc}", f"{paper_val:.4f}", "—"))

    headers = ["system", "rate(req/s)", "reproduced(s/tok)", "paper(s/tok)", "deviation"]
    print_table(rows, headers)

    if not any_found:
        print("\n[No result files found. Run scripts/run_reproduction.sh --fig 6c first.]")


def compare_fig10a(results_dir: str):
    """
    Compare reproduced fig10a relative throughput against paper values.
    Result file pattern:
        base: {results_dir}/base-2000-{input_len}-{output_len}-tp.json
        ours: {results_dir}/ours-2000-{input_len}-{output_len}-tp.json
    """
    print("\n" + "=" * 70)
    print("FIGURE 10a — Throughput Sensitivity (relative throughput: ours/base)")
    print("Synthetic workload, A10G + Llama-3-8B, x16large configuration")
    print("NOTE: values at output_lens 50–300 are ESTIMATED from paper figure.")
    print("      Value at output_len=400 (1.793) is CONFIRMED from README.")
    print("=" * 70)

    # These parameters must match reproduce-fig10a.py exactly
    num_data = 2000
    input_len = 1000
    interv = (0.3, 0.7)   # trim outer 30% warmup/cooldown (same as illustrator.py)

    any_found = False
    rows = []

    for output_len, paper_val in sorted(PAPER_FIG10A.items()):
        fbase = os.path.join(results_dir, f"base-{num_data}-{input_len}-{output_len}-tp.json")
        fours = os.path.join(results_dir, f"ours-{num_data}-{input_len}-{output_len}-tp.json")

        if not os.path.exists(fbase) or not os.path.exists(fours):
            missing = []
            if not os.path.exists(fbase):
                missing.append(os.path.basename(fbase))
            if not os.path.exists(fours):
                missing.append(os.path.basename(fours))
            rows.append((str(output_len), "—  (missing: " + ", ".join(missing) + ")", f"{paper_val:.3f}", "—"))
            continue

        any_found = True
        try:
            reproduced = get_tp(fbase, fours, interv)
            dev = pct_dev(reproduced, paper_val)
            rows.append((
                str(output_len),
                f"{reproduced:.3f}",
                f"{paper_val:.3f}",
                f"{dev:+.1f}%",
            ))
        except Exception as exc:
            rows.append((str(output_len), f"ERROR: {exc}", f"{paper_val:.3f}", "—"))

    headers = ["output_len", "reproduced(rel_tp)", "paper(rel_tp)", "deviation"]
    print_table(rows, headers)

    if not any_found:
        print("\n[No result files found. Run scripts/run_reproduction.sh --fig 10a first.]")


def main():
    parser = argparse.ArgumentParser(
        description="Compare NEO reproduction results against paper figures."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Path to evaluation/results/ directory. Default: <repo_root>/evaluation/results/",
    )
    parser.add_argument(
        "--fig",
        choices=["6c", "10a", "both"],
        default="both",
        help="Which figure to compare (default: both)",
    )
    args = parser.parse_args()

    # Resolve results directory
    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        # Walk up from this script to find repo root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(script_dir)
        results_dir = os.path.join(repo_dir, "evaluation", "results")

    if not os.path.isdir(results_dir):
        print(f"WARNING: results directory does not exist: {results_dir}")
        print("Run scripts/run_reproduction.sh first to generate result files.")
        # Don't exit — still print the tables with "file missing" rows so the
        # user can see what files are expected.
        os.makedirs(results_dir, exist_ok=True)

    print(f"\nResults directory: {results_dir}")
    print(f"Comparing: fig {args.fig}")

    if args.fig in ("6c", "both"):
        compare_fig6c(results_dir)

    if args.fig in ("10a", "both"):
        compare_fig10a(results_dir)

    print("\nDone.")
    print("TODO(gpu-verify): Replace ESTIMATED values in PAPER_FIG6C and")
    print("PAPER_FIG10A (in this script) with values read from the PDF figures")
    print("once you have access to the paper. Confirmed values are marked # CONFIRMED.")


if __name__ == "__main__":
    main()
