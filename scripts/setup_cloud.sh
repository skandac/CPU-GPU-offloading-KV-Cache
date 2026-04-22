#!/usr/bin/env bash
# =============================================================================
# scripts/setup_cloud.sh
#
# End-to-end setup script for NEO on a fresh RunPod A10G instance
# (Ubuntu 22.04 + CUDA 12.1).
#
# USAGE:
#   bash scripts/setup_cloud.sh [--model {a10g|t4|both}] [--hf-token TOKEN]
#
# FLAGS:
#   --model   Which pacpu library (and weights) to set up.
#             a10g  → llama3_8b  TP=1  (fig10a, default)
#             t4    → llama2_7b  TP=1  (fig6c)
#             both  → build both libs and download both weight sets
#   --hf-token  Your Hugging Face token (needed for weight download).
#               Alternatively, export HF_TOKEN before running.
#
# REQUIREMENTS:
#   - Run as root or a user with sudo access
#   - Internet access for apt, snap, pip, and HuggingFace
#   - CUDA 12.1 toolkit already installed (verify: nvcc --version)
#   - At least 50 GB free disk space; 120 GB RAM for A10G full swap space
#
# DO NOT RUN ON THE M4 MAC — this script targets Linux x86-64 only.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "==== $* ===="; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODEL_TARGET="a10g"   # default
HF_TOKEN="${HF_TOKEN:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_TARGET="$2"; shift 2 ;;
        --hf-token)
            HF_TOKEN="$2"; shift 2 ;;
        *)
            die "Unknown argument: $1" ;;
    esac
done

case "$MODEL_TARGET" in
    a10g|t4|both) ;;
    *) die "--model must be one of: a10g, t4, both" ;;
esac

# ---------------------------------------------------------------------------
# Locate the repo root (the directory containing this script's parent)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
log "Repo root: $REPO_DIR"

# ---------------------------------------------------------------------------
# Step 1 — System packages
# ---------------------------------------------------------------------------
step "System packages"

sudo apt-get update -y
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    numactl \
    libomp-dev \
    snapd \
    gcc-11 g++-11 \
    gcc-13 g++-13

# Verify compilers
g++-11 --version | head -1
g++-13 --version | head -1
log "Both g++ compilers installed."

# ---------------------------------------------------------------------------
# Step 2 — Intel ISPC 1.23
# ---------------------------------------------------------------------------
step "Intel ISPC 1.23"

if command -v ispc &>/dev/null && ispc --version 2>&1 | grep -q "1\.23"; then
    log "ISPC 1.23 already installed, skipping."
else
    # Try snap first (preferred on Ubuntu 22.04)
    if command -v snap &>/dev/null; then
        # Ensure snapd socket is running
        sudo systemctl enable --now snapd.socket || true
        sudo snap install ispc --channel latest/edge
        # snap binaries live in /snap/bin — add to PATH for this session
        export PATH="/snap/bin:$PATH"
    else
        # Fallback: download the ISPC release tarball
        log "snap unavailable, downloading ISPC 1.23 tarball..."
        ISPC_URL="https://github.com/ispc/ispc/releases/download/v1.23.0/ispc-v1.23.0-linux.tar.gz"
        wget -q "$ISPC_URL" -O /tmp/ispc.tar.gz
        tar xzf /tmp/ispc.tar.gz -C /tmp
        sudo cp /tmp/ispc-v1.23.0-linux/bin/ispc /usr/local/bin/
        rm -rf /tmp/ispc.tar.gz /tmp/ispc-v1.23.0-linux
    fi
    ispc --version | head -1
fi

# TODO(gpu-verify): if ispc --version reports something other than 1.23.x
#                  the pacpu ISPC kernel may fail to compile.

# ---------------------------------------------------------------------------
# Step 3 — Miniconda + Python 3.10 environment
# ---------------------------------------------------------------------------
step "Miniconda + Python 3.10"

CONDA_PREFIX="${HOME}/miniconda"
if [[ ! -d "$CONDA_PREFIX" ]]; then
    wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_PREFIX"
    rm /tmp/miniconda.sh
    log "Miniconda installed at $CONDA_PREFIX"
else
    log "Miniconda already present at $CONDA_PREFIX"
fi

# Make conda available in this shell without interactive init
source "$CONDA_PREFIX/etc/profile.d/conda.sh"

if conda env list | grep -q "^neo "; then
    log "Conda env 'neo' already exists, skipping create."
else
    conda create -n neo python=3.10 -y
fi

conda activate neo
log "Python: $(python --version)"

# ---------------------------------------------------------------------------
# Step 4 — PyTorch 2.4 + CUDA 12.1
# ---------------------------------------------------------------------------
step "PyTorch 2.4 (CUDA 12.1 wheel)"

# Check if torch is already installed with the right CUDA
if python -c "import torch; assert torch.version.cuda == '12.1'" 2>/dev/null; then
    log "PyTorch 2.4 + CUDA 12.1 already installed, skipping."
else
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121
fi

# Sanity check — CUDA must be reachable
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — check driver/toolkit'
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}')
"
# TODO(gpu-verify): if this assertion fires, check nvcc --version and ensure
#                  the CUDA 12.1 driver is installed.

# ---------------------------------------------------------------------------
# Step 5 — NEO Python dependencies (requirements.txt)
# ---------------------------------------------------------------------------
step "NEO Python dependencies"

cd "$REPO_DIR"
pip install -r requirements.txt
# Installs: fastapi, ray[default], safetensors, transformers,
#           uvicorn, vllm_flash_attn, matplotlib

# ---------------------------------------------------------------------------
# Step 6 — vLLM (needed for vLLM baseline evaluation)
# ---------------------------------------------------------------------------
step "vLLM baseline"

# Pin to 0.5.4 to match the flags used in evaluation/server.py:
#   --disable-async-output-proc, --disable-frontend-multiprocessing,
#   --disable-custom-all-reduce, --tokenizer-pool-size, etc.
# TODO(gpu-verify): if vLLM 0.5.4 wheel is unavailable for CUDA 12.1,
#                  try 0.5.3 or the latest 0.5.x.
pip install vllm==0.5.4

# ---------------------------------------------------------------------------
# Step 7 — SwiftLLM Python package
# ---------------------------------------------------------------------------
step "SwiftLLM package (pip install -e .)"

cd "$REPO_DIR"
pip install -e .
python -c "import swiftllm; print('swiftllm import OK')"

# ---------------------------------------------------------------------------
# Step 8 — CUDA extension (swiftllm_c)
# ---------------------------------------------------------------------------
step "CUDA extension (csrc/)"

# A10G = compute capability 8.6
# The CUDAExtension in csrc/setup.py uses -O3 and --use_fast_math.
# TORCH_CUDA_ARCH_LIST scopes compilation to A10G only (speeds up build).
export TORCH_CUDA_ARCH_LIST="8.6"
# TODO(gpu-verify): for T4 change to "7.5"; for multi-GPU setups add both
#                  e.g. TORCH_CUDA_ARCH_LIST="7.5;8.6"

cd "$REPO_DIR"
pip install -e csrc
python -c "import swiftllm_c; print('swiftllm_c import OK')"

# ---------------------------------------------------------------------------
# Step 9 — pacpu CPU attention library
# ---------------------------------------------------------------------------
step "pacpu CPU attention library"

cd "$REPO_DIR/pacpu"

build_pacpu() {
    local model_name="$1"   # e.g. llama3_8b
    local tp_degree="$2"    # e.g. 1
    local so_name="libpacpu-${model_name}-tp${tp_degree}.so"

    if [[ -f "build/$so_name" ]]; then
        log "$so_name already exists, skipping build."
        return
    fi

    log "Building $so_name ..."
    # build.sh uses:
    #   CUDA_HOST_COMPILER = g++-11   (NVCC rejects g++ > 12)
    #   CXX_COMPILER       = g++-13   (main C++ compiler)
    # Both paths are resolved with `which` inside build.sh.
    bash build.sh "$model_name" "$tp_degree"

    [[ -f "build/$so_name" ]] || die "Build succeeded but $so_name not found"
    log "Built: pacpu/build/$so_name"
}

case "$MODEL_TARGET" in
    a10g) build_pacpu llama3_8b 1 ;;
    t4)   build_pacpu llama2_7b 1 ;;
    both)
        build_pacpu llama3_8b 1
        build_pacpu llama2_7b 1
        ;;
esac

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Step 10 — Model weights download
# ---------------------------------------------------------------------------
step "Model weights"

WEIGHTS_DIR="${HOME}/weights"
mkdir -p "$WEIGHTS_DIR"

if [[ -z "$HF_TOKEN" ]]; then
    log "WARNING: --hf-token not provided and HF_TOKEN env var is unset."
    log "         Skipping weight download. Set HF_TOKEN and re-run, or"
    log "         download manually with huggingface-cli."
else
    pip install -q 'huggingface_hub[cli]'
    huggingface-cli login --token "$HF_TOKEN"

    download_weights() {
        local repo="$1"         # e.g. meta-llama/Llama-3.1-8B
        local local_dir="$2"    # e.g. /root/weights/Llama-3-8B
        if [[ -d "$local_dir" && "$(ls -A "$local_dir" 2>/dev/null)" ]]; then
            log "Weights already present at $local_dir, skipping."
            return
        fi
        log "Downloading $repo → $local_dir ..."
        huggingface-cli download "$repo" \
            --local-dir "$local_dir" \
            --exclude "*.pth"
    }

    case "$MODEL_TARGET" in
        a10g)
            download_weights meta-llama/Llama-3.1-8B "$WEIGHTS_DIR/Llama-3-8B"
            ;;
        t4)
            download_weights meta-llama/Llama-2-7b-hf "$WEIGHTS_DIR/Llama-2-7b-hf"
            ;;
        both)
            download_weights meta-llama/Llama-3.1-8B  "$WEIGHTS_DIR/Llama-3-8B"
            download_weights meta-llama/Llama-2-7b-hf "$WEIGHTS_DIR/Llama-2-7b-hf"
            ;;
    esac
fi

# ---------------------------------------------------------------------------
# Step 11 — Patch evaluation configs with actual weight paths
# ---------------------------------------------------------------------------
step "Patching evaluation configs"

# RunPod default user is root; adjust USER_HOME if your instance uses ubuntu.
USER_HOME="${HOME}"

patch_config() {
    local config_file="$1"
    local model_dir="$2"

    if [[ ! -f "$config_file" ]]; then
        log "WARNING: $config_file not found, skipping patch."
        return
    fi

    # Use Python for JSON-safe in-place edit (avoids sed quoting pitfalls)
    python - "$config_file" "$model_dir" <<'PYEOF'
import sys, json
cfg_path, model_dir = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    cfg = json.load(f)
cfg["model_path"] = model_dir
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=4)
print(f"  Patched model_path → {model_dir} in {cfg_path}")
PYEOF
}

case "$MODEL_TARGET" in
    a10g)
        patch_config "$REPO_DIR/evaluation/configs/config-a10-8b.json" \
                     "$USER_HOME/weights/Llama-3-8B"
        ;;
    t4)
        patch_config "$REPO_DIR/evaluation/configs/config-t4-7b.json" \
                     "$USER_HOME/weights/Llama-2-7b-hf"
        ;;
    both)
        patch_config "$REPO_DIR/evaluation/configs/config-a10-8b.json" \
                     "$USER_HOME/weights/Llama-3-8B"
        patch_config "$REPO_DIR/evaluation/configs/config-t4-7b.json" \
                     "$USER_HOME/weights/Llama-2-7b-hf"
        ;;
esac

# ---------------------------------------------------------------------------
# Step 12 — Runtime directories
# ---------------------------------------------------------------------------
step "Runtime directories"

mkdir -p "$REPO_DIR/profile_results"
# The NEO server writes latency-profiling JSON here.
# Path is hard-coded in evaluation/server.py:
#   {neo_dir}/profile_results/

mkdir -p "$REPO_DIR/docs/figures"
# Used by scripts/run_reproduction.sh to save output PDFs.

log "Runtime directories created."

# ---------------------------------------------------------------------------
# Step 13 — Smoke test
# ---------------------------------------------------------------------------
step "Smoke test"

# Only run if weights are present
SMOKE_WEIGHTS=""
case "$MODEL_TARGET" in
    a10g|both) SMOKE_WEIGHTS="$USER_HOME/weights/Llama-3-8B" ;;
    t4)        SMOKE_WEIGHTS="$USER_HOME/weights/Llama-2-7b-hf" ;;
esac
SMOKE_MODEL=""
case "$MODEL_TARGET" in
    a10g|both) SMOKE_MODEL="llama3_8b" ;;
    t4)        SMOKE_MODEL="llama2_7b" ;;
esac

if [[ -d "$SMOKE_WEIGHTS" && "$(ls -A "$SMOKE_WEIGHTS" 2>/dev/null)" ]]; then
    log "Running offline smoke test..."
    cd "$REPO_DIR"
    python examples/example.py \
        --model-path "$SMOKE_WEIGHTS" \
        --model-name "$SMOKE_MODEL" \
        --num-gpu-blocks 50 \
        --swap-space 2 \
        --prompt-path examples/example.txt \
        --profile-result-path "$REPO_DIR/profile_results/"
    log "Smoke test completed."
else
    log "Weights not downloaded yet — skipping smoke test."
    log "Run after downloading weights:"
    log "  python examples/example.py \\"
    log "    --model-path $SMOKE_WEIGHTS \\"
    log "    --model-name $SMOKE_MODEL \\"
    log "    --num-gpu-blocks 50 --swap-space 2 \\"
    log "    --prompt-path examples/example.txt \\"
    log "    --profile-result-path $REPO_DIR/profile_results/"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
step "Setup complete"
cat <<EOF

All steps completed. Summary of installed artifacts:

  Python env     : $CONDA_PREFIX/envs/neo
  swiftllm       : $REPO_DIR (editable install)
  swiftllm_c     : $REPO_DIR/csrc (CUDA extension)
  pacpu libs     : $REPO_DIR/pacpu/build/
  Profile dir    : $REPO_DIR/profile_results/
  Figures dir    : $REPO_DIR/docs/figures/

Next steps:
  1. Ensure model weights are at ~/weights/Llama-3-8B (or Llama-2-7b-hf)
  2. conda activate neo
  3. cd $REPO_DIR/evaluation
  4. python reproduce-fig10a.py     # A10G — ~5 h with 2000 requests
     python reproduce-fig6c.py      # T4 — ~30 min with 100 requests

For detailed notes see: docs/setup-notes.md
EOF
