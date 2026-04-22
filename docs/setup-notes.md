# NEO Setup Notes — Ubuntu 22.04 + CUDA 12.1 + RunPod A10G

This document covers a fresh install of NEO on a RunPod `gpu_1x_A10G` (or any
bare Ubuntu 22.04 node with an NVIDIA A10G and CUDA 12.1).

For a T4 instance (fig6c reproduction) see the "T4 / Llama-2-7B differences"
section at the bottom; the steps are identical except where noted.

---

## 0. Assumed starting state

| Item | Expected value |
|------|----------------|
| OS   | Ubuntu 22.04 LTS |
| CUDA toolkit | 12.1 (pre-installed on RunPod images; verify with `nvcc --version`) |
| GPU  | NVIDIA A10G (24 GB) |
| RAM  | ≥ 120 GB (for full 120 GB CPU KV-cache swap space) |
| Python | not yet installed (we install via Miniconda) |
| Storage | ≥ 50 GB free in the working directory |

RunPod A10G pods typically ship CUDA 12.1 drivers. If `nvcc --version` shows a
different version, adjust the PyTorch wheel URL in step 3 accordingly.

---

## 1. System packages

```bash
sudo apt-get update && sudo apt-get install -y \
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

# Confirm both compilers are present
g++-11 --version   # should print 11.x
g++-13 --version   # should print 13.x
```

**Why two g++ versions?**
- `g++-13` is the primary C++ compiler used by the pacpu CMake build
  (`-march=native`, OpenMP, C++17).
- `g++-11` is passed to NVCC as `CUDA_HOST_COMPILER` because NVCC on CUDA 12.1
  rejects host compilers newer than GCC 12. See `pacpu/build.sh` line 2.

---

## 2. Intel ISPC 1.23 (required for pacpu CPU kernel)

The pacpu library uses ISPC for auto-vectorised SIMD code on x86-64 with
AVX2/AVX-512. Version **1.23** is required; other versions may differ in ABI.

```bash
# snapd must be running (it is by default on Ubuntu 22.04)
sudo systemctl enable --now snapd.socket
sudo snap install ispc --channel latest/edge
# Verify
ispc --version   # must print 1.23.x
```

If `snap` is unavailable (e.g., minimal cloud image), install from the ISPC
GitHub releases page instead:
```bash
wget https://github.com/ispc/ispc/releases/download/v1.23.0/ispc-v1.23.0-linux.tar.gz
tar xzf ispc-v1.23.0-linux.tar.gz
sudo cp ispc-v1.23.0-linux/bin/ispc /usr/local/bin/
```

---

## 3. Python environment (Miniconda)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda
# Add to PATH for this session
export PATH="$HOME/miniconda/bin:$PATH"

conda create -n neo python=3.10 -y
conda activate neo
```

---

## 4. PyTorch 2.4 + CUDA 12.1

```bash
# Official PyTorch wheel for CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121
# Quick sanity check (must print True)
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 5. NEO Python dependencies

```bash
cd ~/NEO   # or wherever the repo lives; adjust to your RunPod path

pip install -r requirements.txt
# Installs: fastapi, ray[default], safetensors, transformers,
#           uvicorn, vllm_flash_attn, matplotlib
```

`vllm_flash_attn` is a custom Flash Attention build distributed by the vLLM
project. It requires CUDA to be present at install time, which it is by step 4.

---

## 6. vLLM (needed for the vLLM baseline in evaluation scripts)

```bash
pip install vllm==0.5.4
# Pin to 0.5.4 to match the CLI flags used in evaluation/server.py
# (--disable-async-output-proc, --disable-frontend-multiprocessing, etc.)
# TODO(gpu-verify): confirm vLLM version against the flags in evaluation/server.py
#                  if a newer vLLM drops these flags, pin lower.
```

---

## 7. SwiftLLM Python package

```bash
cd ~/NEO
pip install -e .
# Installs the `swiftllm` package in editable mode.
# Verify:
python -c "import swiftllm; print('swiftllm OK')"
```

---

## 8. CUDA extension (swiftllm_c)

```bash
cd ~/NEO
pip install -e csrc
# Compiles: block_swapping.cpp, small_kernels.cu, linear.cu
# (attention.cu is commented out — flash attn used instead)
# Output: swiftllm_c.cpython-310-x86_64-linux-gnu.so in csrc/
# TODO(gpu-verify): if nvcc errors on 'unsupported GPU arch', add
#                  TORCH_CUDA_ARCH_LIST="8.6" before pip install -e csrc
#                  (8.6 = A10G compute capability)
```

---

## 9. pacpu CPU attention library

This step must be run once per (model, tensor-parallel-degree) pair. The
output is a `.so` loaded at runtime by the NEO engine.

```bash
cd ~/NEO/pacpu

# Primary: Llama-3-8B, TP=1 (for fig10a / A10G reproduction)
bash build.sh llama3_8b 1
# Output: build/libpacpu-llama3_8b-tp1.so
# TODO(gpu-verify): if cmake cannot find Torch, ensure the conda env is
#                  active so `python -c 'import torch'` works.

# Secondary: Llama-2-7B, TP=1 (for fig6c / T4 reproduction; skip on A10G)
# bash build.sh llama2_7b 1
# Output: build/libpacpu-llama2_7b-tp1.so

cd ~/NEO
```

---

## 10. Model weights

### A10G — Llama-3-8B (Llama-3.1-8B)

```bash
pip install 'huggingface_hub[cli]'
huggingface-cli login --token <YOUR_HF_TOKEN>

mkdir -p ~/weights
huggingface-cli download meta-llama/Llama-3.1-8B \
    --local-dir ~/weights/Llama-3-8B \
    --exclude "*.pth"
# ~16 GB download
```

### T4 — Llama-2-7B (only for fig6c)

```bash
huggingface-cli download meta-llama/Llama-2-7b-hf \
    --local-dir ~/weights/Llama-2-7b-hf \
    --exclude "*.pth"
# ~13 GB download
```

---

## 11. Edit evaluation configs

Update `model_path` in the relevant config to point to your weight directory.

**A10G (fig10a):**
```bash
# edit evaluation/configs/config-a10-8b.json
# change: "model_path": "/home/ubuntu/weights/Llama-3-8B"
# to:     "model_path": "/root/weights/Llama-3-8B"   (RunPod default user is root)
```

**T4 (fig6c):**
```bash
# edit evaluation/configs/config-t4-7b.json
# change: "model_path": "/home/ubuntu/weights/Llama-2-7b-hf"
# to your actual path
```

---

## 12. Create required runtime directories

```bash
mkdir -p ~/NEO/profile_results
# The NEO server writes profiling JSON here; the path is hardcoded in
# evaluation/server.py as {repo_dir}/profile_results/
```

---

## 13. Smoke test

```bash
cd ~/NEO
python examples/example.py \
    --model-path ~/weights/Llama-3-8B \
    --model-name llama3_8b \
    --num-gpu-blocks 50 \
    --swap-space 2 \
    --prompt-path examples/example.txt

# Expected: prints generated text + throughput numbers.
# If you see CUDA OOM, reduce --num-gpu-blocks.
# TODO(gpu-verify): run this before attempting full reproduction.
```

---

## 14. Full reproduction (after smoke test passes)

```bash
cd ~/NEO

# Fig 6c — load-latency curve (T4 + Llama-2-7B, ~30 min with 100 requests)
cd evaluation && python reproduce-fig6c.py
# Output: evaluation/fig6c.pdf

# Fig 10a — throughput sensitivity (A10G + Llama-3-8B, ~5 h with 2000 requests)
cd evaluation && python reproduce-fig10a.py
# Output: evaluation/fig10a.pdf
```

---

## T4 / Llama-2-7B differences

| Step | A10G change | T4 value |
|------|-------------|----------|
| GPU  | A10G        | Tesla T4 |
| RAM  | ≥ 120 GB    | ≥ 30 GB  |
| Model | Llama-3.1-8B | Llama-2-7b-hf |
| pacpu build | `bash build.sh llama3_8b 1` | `bash build.sh llama2_7b 1` |
| Config file | `config-a10-8b.json` | `config-t4-7b.json` |
| Repro script | `reproduce-fig10a.py` | `reproduce-fig6c.py` |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `nvcc: error: unsupported gpu architecture` during `pip install -e csrc` | `export TORCH_CUDA_ARCH_LIST="8.6"` then retry |
| `ispc: command not found` after snap install | `export PATH=/snap/bin:$PATH` |
| cmake can't find Torch | Make sure the conda env is active; run `python -c 'import torch; print(torch.utils.cmake_prefix_path)'` |
| Server never prints "Started server process" | Check `evaluation/<name>-server.log` for the actual error |
| `libpacpu-llama3_8b-tp1.so` not found at runtime | Verify `pacpu/build/` exists and the filename matches `config-a10-8b.json`'s `library` field |
| `numactl` not found | `sudo apt-get install -y numactl` |
