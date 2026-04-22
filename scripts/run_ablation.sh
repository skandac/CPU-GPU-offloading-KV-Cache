#!/usr/bin/env bash
# =============================================================================
# scripts/run_ablation.sh
#
# M10 quantization-granularity ablation. Sweeps:
#     granularity ∈ {per-token, per-channel, per-token-per-head}
# across the M8 benchmarks (throughput, latency, perplexity) for INT8 variants.
#
# USAGE (from repo root, with conda env 'neo' active):
#     bash scripts/run_ablation.sh [--variant int8-transfer|int8-cpu-kv|both]
#                                  [--skip perplexity|throughput|latency|none]
#                                  [--config evaluation/configs/config-a10-8b.json]
#                                  [--n 200]
#
# OUTPUTS (in benchmarks/results/):
#     throughput_{variant}_{granularity}_{workload}.csv
#     latency_{variant}_{granularity}_{workload}_r{rate}.csv
#     perplexity_{variant}_{granularity}.csv
#
# The ablation_summary.csv at the end rolls all granularities into one table
# for easy inclusion in docs/results.md (M11).
#
# DO NOT RUN ON A LAPTOP. Requires CUDA + NEO install.
# =============================================================================

set -euo pipefail

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "==== $* ===="; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
VARIANT="int8-transfer"
SKIP=""
NREQS=200
RATE=1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$REPO_DIR/evaluation/configs/config-a10-8b.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --skip)    SKIP="$2"; shift 2 ;;
        --config)  CONFIG="$2"; shift 2 ;;
        --n)       NREQS="$2"; shift 2 ;;
        --rate)    RATE="$2"; shift 2 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

case "$VARIANT" in
    int8-transfer|int8-cpu-kv|both) ;;
    *) die "--variant must be one of: int8-transfer, int8-cpu-kv, both" ;;
esac

PYTHON="${PYTHON:-python}"
BENCH_DIR="$REPO_DIR/benchmarks"
OUT_DIR="$BENCH_DIR/results"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Which variants?
# ---------------------------------------------------------------------------
if [[ "$VARIANT" == "both" ]]; then
    VARIANTS_TO_RUN=("int8-transfer" "int8-cpu-kv")
else
    VARIANTS_TO_RUN=("$VARIANT")
fi

GRANULARITIES=("per-token" "per-channel" "per-token-per-head")

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
for v in "${VARIANTS_TO_RUN[@]}"; do
    step "Variant: $v"

    for g in "${GRANULARITIES[@]}"; do
        # int8-cpu-kv cannot honor per-channel granularity — the online
        # append path would have to re-quantize a whole block on every
        # token. See swiftllm/worker/block_swapper.py:__init__ and
        # docs/int8-design.md §3, §5. Skip that cell here rather than
        # letting the engine fail-fast mid-sweep.
        if [[ "$v" == "int8-cpu-kv" && "$g" == "per-channel" ]]; then
            log "  [skip] $v / $g — unsupported combination (see block_swapper guard)"
            continue
        fi

        step "  Granularity: $g"

        if [[ "$SKIP" != "throughput" ]]; then
            log "  throughput: $v / $g"
            "$PYTHON" "$BENCH_DIR/run_throughput.py" \
                --variant "$v" \
                --quant-granularity "$g" \
                --config "$CONFIG" \
                --num-requests "$NREQS" \
                --workloads azure-code,osc,synthetic \
                --out "$OUT_DIR"
        fi

        if [[ "$SKIP" != "latency" ]]; then
            log "  latency: $v / $g"
            "$PYTHON" "$BENCH_DIR/run_latency.py" \
                --variant "$v" \
                --quant-granularity "$g" \
                --workload azure-code \
                --rate "$RATE" \
                --num-requests "$NREQS" \
                --config "$CONFIG" \
                --out "$OUT_DIR"
        fi

        if [[ "$SKIP" != "perplexity" ]]; then
            log "  perplexity: $v / $g"
            MODEL_PATH="$("$PYTHON" -c "import json; print(json.load(open('$CONFIG'))['model_path'])")"
            "$PYTHON" "$BENCH_DIR/run_perplexity.py" \
                --variant "$v" \
                --quant-granularity "$g" \
                --model-path "$MODEL_PATH" \
                --out "$OUT_DIR"
        fi
    done
done

# ---------------------------------------------------------------------------
# Roll-up: one ablation_summary.csv per variant
# ---------------------------------------------------------------------------
step "Building ablation summary"

"$PYTHON" - <<'PY'
import csv, glob, os, json, re

here = os.environ.get("OUT_DIR") or "benchmarks/results"
rows = []

# perplexity
for f in glob.glob(os.path.join(here, "perplexity_int8-*.csv")):
    with open(f) as fp:
        r = list(csv.DictReader(fp))
    if not r:
        continue
    row = r[0]
    m = re.match(r"perplexity_(int8-[a-z-]+)_(per-[a-z-]+)\.csv", os.path.basename(f))
    if not m:
        continue
    variant, gran = m.groups()
    rows.append({"variant": variant, "granularity": gran,
                 "metric": "perplexity", "value": row["perplexity"]})

# throughput — average effective_throughput across workloads/rates
for f in glob.glob(os.path.join(here, "throughput_int8-*.csv")):
    m = re.match(r"throughput_(int8-[a-z-]+)_(per-[a-z-]+)_([a-z-]+)\.csv", os.path.basename(f))
    if not m:
        continue
    variant, gran, wl = m.groups()
    with open(f) as fp:
        rs = list(csv.DictReader(fp))
    if not rs:
        continue
    avg = sum(float(r["effective_throughput_req_s"]) for r in rs) / len(rs)
    rows.append({"variant": variant, "granularity": gran,
                 "metric": f"avg_throughput_{wl}", "value": f"{avg:.3f}"})

# latency — per-token p99
for f in glob.glob(os.path.join(here, "latency_int8-*.csv")):
    m = re.match(r"latency_(int8-[a-z-]+)_(per-[a-z-]+)_([a-z-]+)_r.*\.csv", os.path.basename(f))
    if not m:
        continue
    variant, gran, wl = m.groups()
    with open(f) as fp:
        rs = list(csv.DictReader(fp))
    if not rs:
        continue
    vals = sorted(float(r["per_token_latency_s"]) for r in rs)
    idx = max(0, int(len(vals) * 0.99) - 1)
    rows.append({"variant": variant, "granularity": gran,
                 "metric": f"per_token_p99_{wl}", "value": f"{vals[idx]:.5f}"})

out_path = os.path.join(here, "ablation_summary.csv")
with open(out_path, "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=["variant", "granularity", "metric", "value"])
    w.writeheader()
    w.writerows(rows)
print(f"[ok] wrote {out_path}")
PY

step "Done"
log "Per-run CSVs in:      $OUT_DIR"
log "Summary:              $OUT_DIR/ablation_summary.csv"
