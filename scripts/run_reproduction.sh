#!/usr/bin/env bash
# =============================================================================
# scripts/run_reproduction.sh
#
# Runs both NEO reproduction scripts and collects outputs into docs/figures/.
#
# USAGE (from repo root):
#   bash scripts/run_reproduction.sh [--fig {6c|10a|both}] [--quick]
#
# FLAGS:
#   --fig     Which figure to reproduce (default: both)
#   --quick   Passed through as an environment hint; the reproduce scripts
#             already use 100 requests for fig6c by default (fast). For fig10a,
#             set NUM_DATA and OUTPUT_LENS env vars (see below) to reduce load.
#
# ENVIRONMENT OVERRIDES (fig10a only):
#   NEO_FIG10A_NUM_DATA    override num_data   (default 2000, paper uses 2000)
#   NEO_FIG10A_OUTPUT_LENS override output_lens as space-separated list
#                          (default "50 100 200 300 400")
#   Example: NEO_FIG10A_NUM_DATA=800 bash scripts/run_reproduction.sh --fig 10a
#
# OUTPUTS:
#   evaluation/fig6c.pdf         → docs/figures/fig6c.pdf
#   evaluation/fig10a.pdf        → docs/figures/fig10a.pdf
#   evaluation/results/*.json    → raw latency / throughput data (untouched)
#   evaluation/evaluation.log    → server start/stop + timing logs
#   evaluation/{ours,vllm,base}-server.log → per-server stderr logs
#
# NOTES:
#   - Must be run from the repo root directory.
#   - The conda env 'neo' must be active, or PYTHON must point to the right
#     interpreter (default: uses whatever `python` resolves to in PATH).
#   - The reproduce scripts start and stop their own server subprocesses on
#     port 8000; make sure no other process holds that port.
#   - fig10a takes ~5 h at full scale (2000 requests × 5 output lengths).
#     Reduce NEO_FIG10A_NUM_DATA to ~800 for a faster (~2 h) run.
#   - TODO(gpu-verify): if the server never reaches "Started server process",
#     tail evaluation/ours-server.log or evaluation/vllm-server.log to debug.
# =============================================================================

set -euo pipefail

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "==== $* ===="; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
FIG_TARGET="both"
QUICK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fig)   FIG_TARGET="$2"; shift 2 ;;
        --quick) QUICK=1; shift ;;
        *) die "Unknown argument: $1" ;;
    esac
done

case "$FIG_TARGET" in
    6c|10a|both) ;;
    *) die "--fig must be one of: 6c, 10a, both" ;;
esac

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_DIR="$REPO_DIR/evaluation"
FIGURES_DIR="$REPO_DIR/docs/figures"

PYTHON="${PYTHON:-python}"   # override with `PYTHON=/path/to/python3` if needed

# Sanity: confirm we're in the right repo
[[ -f "$REPO_DIR/setup.py" ]] || die "Not in NEO repo root. Run from $REPO_DIR."
[[ -d "$EVAL_DIR" ]] || die "evaluation/ directory not found."

# Output directory for collected figures
mkdir -p "$FIGURES_DIR"

# ---------------------------------------------------------------------------
# Check port 8000 is free before starting
# ---------------------------------------------------------------------------
check_port() {
    if ss -tlnp 2>/dev/null | grep -q ':8000 ' || \
       netstat -tlnp 2>/dev/null | grep -q ':8000 '; then
        die "Port 8000 is already in use. Kill the occupying process first."
    fi
}

# ---------------------------------------------------------------------------
# fig6c: load-latency curve (T4, Llama-2-7B, OpenAI summarization)
# Servers: vllm, ours
# Result files: evaluation/results/{vllm,ours}-osc-lat-*.json
# Output PDF: evaluation/fig6c.pdf
# Estimated time: ~30 min (100 requests, 11 rate points total)
# ---------------------------------------------------------------------------
run_fig6c() {
    step "Figure 6c — load-latency curve"
    check_port

    # The script runs two server rounds (vllm then ours), each across multiple
    # request rates. Results are cached to evaluation/results/ — re-running
    # with the same config reuses cached files (see benchmark.py:run_test).
    log "Starting reproduce-fig6c.py (expected ~30 min at 100 requests)..."
    log "Tip: tail -f $EVAL_DIR/evaluation.log for progress."

    cd "$EVAL_DIR"
    "$PYTHON" reproduce-fig6c.py
    cd "$REPO_DIR"

    # Collect the PDF
    if [[ -f "$EVAL_DIR/fig6c.pdf" ]]; then
        cp "$EVAL_DIR/fig6c.pdf" "$FIGURES_DIR/fig6c.pdf"
        log "Saved: docs/figures/fig6c.pdf"
    else
        log "WARNING: fig6c.pdf not found in evaluation/ — check evaluation.log"
    fi
}

# ---------------------------------------------------------------------------
# fig10a: throughput sensitivity (A10G, Llama-3-8B, synthetic workloads)
# Servers: base, ours
# Result files: evaluation/results/{base,ours}-2000-1000-{50..400}-tp.json
# Output PDF: evaluation/fig10a.pdf
# Estimated time: ~5 h at full scale (num_data=2000, 5 output lengths)
# ---------------------------------------------------------------------------
run_fig10a() {
    step "Figure 10a — throughput sensitivity"
    check_port

    # Allow lightweight override for quick iteration on the GPU:
    #   NEO_FIG10A_NUM_DATA=800 NEO_FIG10A_OUTPUT_LENS="100 200 400" \
    #     bash scripts/run_reproduction.sh --fig 10a
    #
    # If the env vars are set we patch the script on-the-fly with a
    # temporary copy so the originals are never modified.
    #
    # TODO(gpu-verify): below-800 num_data degrades throughput accuracy due to
    #                   warm-up/cool-down effects (see paper Fig 10a caption).
    NUM_DATA="${NEO_FIG10A_NUM_DATA:-}"
    OUTPUT_LENS="${NEO_FIG10A_OUTPUT_LENS:-}"

    if [[ -n "$NUM_DATA" || -n "$OUTPUT_LENS" ]]; then
        TMPSCRIPT="$(mktemp /tmp/reproduce-fig10a-XXXX.py)"
        cp "$EVAL_DIR/reproduce-fig10a.py" "$TMPSCRIPT"

        if [[ -n "$NUM_DATA" ]]; then
            sed -i "s/^num_data = .*/num_data = $NUM_DATA/" "$TMPSCRIPT"
            log "Overriding num_data → $NUM_DATA"
        fi
        if [[ -n "$OUTPUT_LENS" ]]; then
            # Convert space-separated list "50 100 200" → Python list "[50, 100, 200]"
            PY_LIST="[$(echo "$OUTPUT_LENS" | tr ' ' ',')]"
            sed -i "s/^output_lens = .*/output_lens = $PY_LIST/" "$TMPSCRIPT"
            log "Overriding output_lens → $PY_LIST"
        fi

        REPRO_SCRIPT="$TMPSCRIPT"
    else
        REPRO_SCRIPT="$EVAL_DIR/reproduce-fig10a.py"
    fi

    log "Starting reproduce-fig10a.py (expected ~5 h at full scale)..."
    log "Tip: tail -f $EVAL_DIR/evaluation.log for progress."

    cd "$EVAL_DIR"
    "$PYTHON" "$REPRO_SCRIPT"
    cd "$REPO_DIR"

    [[ -n "$NUM_DATA" || -n "$OUTPUT_LENS" ]] && rm -f "$TMPSCRIPT"

    # Collect the PDF
    if [[ -f "$EVAL_DIR/fig10a.pdf" ]]; then
        cp "$EVAL_DIR/fig10a.pdf" "$FIGURES_DIR/fig10a.pdf"
        log "Saved: docs/figures/fig10a.pdf"
    else
        log "WARNING: fig10a.pdf not found in evaluation/ — check evaluation.log"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
log "NEO reproduction runner"
log "Repo: $REPO_DIR"
log "Python: $("$PYTHON" --version 2>&1)"
log "Target: fig $FIG_TARGET"

case "$FIG_TARGET" in
    6c)   run_fig6c ;;
    10a)  run_fig10a ;;
    both) run_fig6c; run_fig10a ;;
esac

step "Done"
log "Figures written to: $FIGURES_DIR"
log "Raw results in: $EVAL_DIR/results/"
log ""
log "To compare against paper numbers:"
log "  python scripts/compare_to_paper.py"
log ""
log "To inspect server logs:"
log "  cat $EVAL_DIR/ours-server.log"
log "  cat $EVAL_DIR/vllm-server.log"
log "  cat $EVAL_DIR/base-server.log"
