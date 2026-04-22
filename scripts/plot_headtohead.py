"""
scripts/plot_headtohead.py

Produces head-to-head NEO-vs-vLLM comparison plots from the CSVs emitted by
scripts/run_vllm_baseline.py.

USAGE (from repo root):
    python scripts/plot_headtohead.py [flags]

FLAGS:
    --csv-dir PATH      Directory containing the CSVs.
                        Default: evaluation/csv_results/
    --out-dir PATH      Output directory for PDFs.
                        Default: docs/figures/
    --workloads LIST    Comma-separated workloads to include in plots.
                        Default: "osc,azure-code"
    --systems LIST      Comma-separated systems to compare.
                        Default: "vllm,ours"

INPUTS (expected filenames in --csv-dir):
    {system}_{workload}_requests.csv
    {system}_{workload}_summary.csv
    For every (system, workload) pair you want in the plots.

    To generate these, run for each system:
        python scripts/run_vllm_baseline.py --system vllm
        python scripts/run_vllm_baseline.py --system ours

OUTPUTS (into --out-dir):
    headtohead_latency_{workload}.pdf    Load-latency curves (all systems)
    headtohead_throughput_{workload}.pdf Throughput vs offered rate bar chart
    headtohead_p99_{workload}.pdf        p99 latency vs offered rate

PLOT SEMANTICS:
    Latency plot mirrors evaluation/illustrator.py:draw_one_rl_diagram — same
    x-axis (request rate) and y-axis (avg per-token latency), overlaid lines
    per system. This is essentially a customizable fig6c.

    Throughput plot shows effective throughput (req/s) at each offered rate.
    When a system saturates, effective < offered. The gap is visual evidence
    of the vLLM-vs-NEO advantage.

DEPENDENCIES: matplotlib, pandas (both in requirements.txt via matplotlib and
              transitively; pandas is pulled in by ray[default]).

TODO(gpu-verify): check that matplotlib + pandas are actually importable in
                  the 'neo' conda env. If pandas is missing, install via
                  `pip install pandas` (tiny dep, no GPU needed).
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

# matplotlib is in requirements.txt
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Defaults — per-system styling for consistent plots across runs
# ---------------------------------------------------------------------------
SYSTEM_STYLES = {
    "vllm": {"label": "vLLM",   "marker": "o", "color": "#1f77b4"},
    "ours": {"label": "NEO",    "marker": "x", "color": "#ff7f0e"},
    "base": {"label": "Base",   "marker": "s", "color": "#2ca02c"},
}


# ---------------------------------------------------------------------------
# CSV loading — no pandas dependency; keep it simple
# ---------------------------------------------------------------------------
def load_summary_csv(path: str) -> list:
    """Return list of dicts, one per row, with float-casted numeric columns."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Cast all numeric cells — skip system/workload strings
            for k in ("rate_req_s", "n_requests", "throughput_req_s",
                      "avg_per_token_latency_s", "p50_latency_s",
                      "p95_latency_s", "p99_latency_s"):
                if k in r and r[k] != "":
                    r[k] = float(r[k])
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Plotting — one PDF per workload, one line per system
# ---------------------------------------------------------------------------
def plot_latency_curve(
    summaries: dict,     # {system: [row dicts]}
    workload: str,
    out_path: str,
) -> None:
    """
    Load-latency curve: x = offered rate (req/s), y = avg per-token latency (s)
    Shape mirrors evaluation/illustrator.py:draw_one_rl_diagram.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    max_rate = 0.0
    max_lat = 0.0
    for system, rows in summaries.items():
        style = SYSTEM_STYLES.get(system, {"label": system, "marker": ".", "color": "gray"})
        rows_sorted = sorted(rows, key=lambda r: r["rate_req_s"])
        xs = [r["rate_req_s"] for r in rows_sorted]
        ys = [r["avg_per_token_latency_s"] for r in rows_sorted]
        if not xs:
            continue
        ax.plot(xs, ys,
                label=style["label"], marker=style["marker"],
                color=style["color"])
        max_rate = max(max_rate, max(xs))
        max_lat  = max(max_lat, max(ys))

    ax.set_xlabel("Request rate (req/s)", fontsize="large")
    ax.set_ylabel("Avg per-token latency (s)", fontsize="large")
    ax.set_title(f"Load-latency — {workload}", fontsize="large")
    ax.grid(True, alpha=0.4)
    ax.legend()
    # Pad axes slightly
    if max_rate > 0:
        ax.set_xlim(0, max_rate * 1.05)
    if max_lat > 0:
        ax.set_ylim(0, max_lat * 1.1)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def plot_throughput_bars(
    summaries: dict,
    workload: str,
    out_path: str,
) -> None:
    """
    Bar chart of effective throughput (req/s) at each offered rate.
    Groups bars by rate, with one bar per system side-by-side.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

    # Collect the union of rates seen across all systems, sorted
    all_rates = sorted({r["rate_req_s"] for rows in summaries.values() for r in rows})
    if not all_rates:
        print(f"[warn] no rates for workload={workload} (skipping throughput plot)")
        plt.close(fig)
        return

    n_sys = len(summaries)
    bar_w = 0.8 / max(n_sys, 1)

    for i, (system, rows) in enumerate(summaries.items()):
        style = SYSTEM_STYLES.get(system, {"label": system, "color": "gray"})
        by_rate = {r["rate_req_s"]: r["throughput_req_s"] for r in rows}
        xs = [j + (i - (n_sys - 1) / 2) * bar_w for j in range(len(all_rates))]
        ys = [by_rate.get(rate, 0.0) for rate in all_rates]
        ax.bar(xs, ys, width=bar_w, label=style["label"], color=style["color"])

    ax.set_xticks(range(len(all_rates)))
    ax.set_xticklabels([f"{r:.2f}" for r in all_rates], fontsize="medium")
    ax.set_xlabel("Offered rate (req/s)", fontsize="large")
    ax.set_ylabel("Effective throughput (req/s)", fontsize="large")
    ax.set_title(f"Throughput — {workload}", fontsize="large")
    ax.grid(True, alpha=0.4, axis="y")
    ax.legend()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def plot_p99_curve(
    summaries: dict,
    workload: str,
    out_path: str,
) -> None:
    """
    p99 end-to-end latency vs offered rate.
    p99 is the most common SLO-related latency metric in serving papers.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    for system, rows in summaries.items():
        style = SYSTEM_STYLES.get(system, {"label": system, "marker": ".", "color": "gray"})
        rows_sorted = sorted(rows, key=lambda r: r["rate_req_s"])
        xs = [r["rate_req_s"] for r in rows_sorted]
        ys = [r["p99_latency_s"] for r in rows_sorted]
        if not xs:
            continue
        ax.plot(xs, ys,
                label=style["label"], marker=style["marker"],
                color=style["color"])

    ax.set_xlabel("Request rate (req/s)", fontsize="large")
    ax.set_ylabel("p99 end-to-end latency (s)", fontsize="large")
    ax.set_title(f"p99 latency — {workload}", fontsize="large")
    ax.grid(True, alpha=0.4)
    ax.legend()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)

    p = argparse.ArgumentParser(description="NEO vs vLLM head-to-head plots")
    p.add_argument("--csv-dir",
                   default=os.path.join(repo_dir, "evaluation", "csv_results"))
    p.add_argument("--out-dir",
                   default=os.path.join(repo_dir, "docs", "figures"))
    p.add_argument("--workloads", default="osc,azure-code")
    p.add_argument("--systems", default="vllm,ours")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    os.makedirs(args.out_dir, exist_ok=True)

    for workload in workloads:
        # Load all summaries for this workload, keyed by system
        summaries = {}
        for system in systems:
            sum_path = os.path.join(args.csv_dir, f"{system}_{workload}_summary.csv")
            if not os.path.exists(sum_path):
                print(f"[warn] missing: {sum_path} — skipping {system} for {workload}")
                continue
            summaries[system] = load_summary_csv(sum_path)

        if not summaries:
            print(f"[warn] no data for workload={workload}, skipping plots")
            continue

        plot_latency_curve(
            summaries, workload,
            os.path.join(args.out_dir, f"headtohead_latency_{workload}.pdf"),
        )
        plot_throughput_bars(
            summaries, workload,
            os.path.join(args.out_dir, f"headtohead_throughput_{workload}.pdf"),
        )
        plot_p99_curve(
            summaries, workload,
            os.path.join(args.out_dir, f"headtohead_p99_{workload}.pdf"),
        )

    print("[done] head-to-head plots in:", args.out_dir)


if __name__ == "__main__":
    main()
