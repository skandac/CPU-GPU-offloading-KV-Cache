# EC2 runbook — NEO baseline → INT8 quantization end-to-end

A single-file walkthrough from a clean AWS account to a fully populated
`~/report/` directory containing every artifact you need for the writeup
and the slide deck. Commands are copy-pasteable.

This runbook assumes:

- You are running the M5 → M6 → M7 → M10 code from this branch
  (int8 design doc, INT8 CPU KV cache, INT8 transfer path, granularity
  ablation, HF-based perplexity adapter).
- You do **not** have the paper's prebuilt AMI — we go the manual-install
  route so the runbook is reproducible anywhere.
- You do **not** have model weights in S3 — we download from
  HuggingFace on the instance.
- You are running Option A (quality evals enabled). The perplexity
  path uses the HF-hook adapter in
  `benchmarks/_quant_hooks.py`; see caveats in `docs/results.md §6`.

> **Cost warning.** `g5.16xlarge` on us-east-1 is ~$4.10/hr on-demand at
> time of writing. Stop the instance the moment you step away. A
> CloudWatch alarm or `aws ec2 stop-instances` on a cron are your
> friends. Nothing in this runbook requires the instance to live
> between sessions — all artifacts land in `~/report` which you
> download at the end.

---

## 0. One-time AWS account prep

If you already have a keypair, security group, and CLI access, skip to §1.

### 0.1 Install and configure the AWS CLI locally

```bash
# macOS
brew install awscli

aws configure
# AWS Access Key ID:     <from IAM console>
# AWS Secret Access Key: <from IAM console>
# Default region name:   us-east-1
# Default output format: json
```

### 0.2 Create a keypair

```bash
aws ec2 create-key-pair \
    --key-name neo-runbook \
    --query 'KeyMaterial' --output text > ~/.ssh/neo-runbook.pem
chmod 400 ~/.ssh/neo-runbook.pem
```

### 0.3 Create a security group that allows inbound SSH from your IP only

```bash
MY_IP=$(curl -s https://checkip.amazonaws.com)
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
         --query 'Vpcs[0].VpcId' --output text)

SG_ID=$(aws ec2 create-security-group \
    --group-name neo-runbook-sg \
    --description "SSH from my IP only" \
    --vpc-id $VPC_ID \
    --query GroupId --output text)

aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp --port 22 \
    --cidr ${MY_IP}/32

echo "SG_ID=$SG_ID"
```

### 0.4 Quota check

`g5.16xlarge` is a G-family instance and counts against the "Running
On-Demand G and VT instances" vCPU quota. You need ≥ 64 vCPUs.

```bash
aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code L-DB2E81BA \
    --query 'Quota.Value' --output text
```

If the value is < 64, request an increase in the AWS console
(Service Quotas → EC2 → "Running On-Demand G and VT instances").
**This can take hours to approve** — do it the day before you plan to
run.

---

## 1. Launch the instance

We use a stock Ubuntu 22.04 AMI and install everything manually.

### 1.1 Find the Canonical Ubuntu 22.04 AMI for your region

```bash
AMI_ID=$(aws ec2 describe-images \
    --owners 099720109477 \
    --filters \
        "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
        "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)
echo "AMI_ID=$AMI_ID"
```

### 1.2 Launch

```bash
SG_ID=<sg-id from step 0.3>

aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type g5.16xlarge \
    --key-name neo-runbook \
    --security-group-ids $SG_ID \
    --block-device-mappings '[{
        "DeviceName":"/dev/sda1",
        "Ebs":{"VolumeSize":500,"VolumeType":"gp3","DeleteOnTermination":true}
    }]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=neo-runbook}]' \
    --count 1 \
    --query 'Instances[0].InstanceId' --output text
```

Wait ~60s, then get the public IP:

```bash
INSTANCE_ID=<from output above>
aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

### 1.3 SSH in

```bash
PUB_IP=<public ip>
ssh -i ~/.ssh/neo-runbook.pem ubuntu@$PUB_IP
```

All remaining commands run **on the instance** unless noted.

---

## 2. System setup

### 2.1 Verify the GPU is an A10G

```bash
nvidia-smi
# Expected: "NVIDIA A10G" with 22-24 GB, driver 535+
```

If `nvidia-smi` is missing, install the driver:

```bash
sudo apt-get update -y
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
# wait ~60s, SSH back in
```

### 2.2 Install build dependencies

```bash
sudo apt-get update -y
sudo apt-get install -y \
    build-essential cmake ninja-build git unzip wget curl \
    software-properties-common \
    python3-venv python3-dev

# GCC 11 and 13 — pacpu/build.sh needs both (§README).
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update -y
sudo apt-get install -y g++-11 g++-13
which g++-11 g++-13
```

### 2.3 Install ISPC 1.23

The README recommends `snap`; that doesn't play well on EC2. Use the
GitHub tarball instead.

```bash
cd /tmp
wget -q https://github.com/ispc/ispc/releases/download/v1.23.0/ispc-v1.23.0-linux.tar.gz
tar xf ispc-v1.23.0-linux.tar.gz
sudo cp ispc-v1.23.0-linux/bin/ispc /usr/local/bin/
ispc --version   # must print 1.23
```

### 2.4 Install Miniconda and create the env

```bash
cd ~
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda init bash
# re-source so `conda` is on PATH
source ~/.bashrc

conda create -y -n neo python=3.10
conda activate neo

# Pin torch to a version known to work with NEO's csrc build. The paper
# uses torch 2.4.x.
pip install --upgrade pip
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

### 2.5 CUDA toolkit

Ubuntu 22.04 EC2 AMIs ship `nvcc` only if the "deep learning" AMI was
picked. You installed stock Ubuntu, so add it:

```bash
sudo apt-get install -y nvidia-cuda-toolkit
nvcc --version
# If nvcc reports a different CUDA version than torch.version.cuda,
# pin pip's CUDA build to match (below).
python -c "import torch; print('torch cuda:', torch.version.cuda)"
```

If `nvcc` is 11.5 but torch was built against 12.1, reinstall torch with
`--index-url https://download.pytorch.org/whl/cu118`. csrc's CMake
needs matching majors.

---

## 3. Clone NEO and install

### 3.1 Clone the branch with M5-M10

```bash
cd ~
git clone https://github.com/<your-fork>/NEO.git
cd NEO
git checkout <branch-with-int8-work>

# Sanity-check that the new files are present.
ls docs/int8-design.md
ls swiftllm/worker/quantize.py
ls benchmarks/_quant_hooks.py
ls scripts/run_ablation.sh
ls tests/test_int8_correctness.py
```

### 3.2 Install Python deps and NEO packages

```bash
conda activate neo
pip install -r requirements.txt
pip install -e .
pip install -e csrc      # CUDA kernels — takes several minutes

# Extra deps for the quality evals (not all are pinned in requirements.txt).
pip install transformers==4.44.0 accelerate==0.33.0
pip install rouge-score datasets
pip install 'huggingface_hub[cli]'
```

### 3.3 Build the CPU operator (pacpu) for Llama-3-8B

```bash
cd ~/NEO/pacpu
bash build.sh llama3_8b 1      # model name + TP degree
cd ~/NEO
ls pacpu/build/libpacpu-llama3_8b-tp1.so   # should exist
```

If you also want to run the 7B fig6c reproduction, build for that too:

```bash
cd ~/NEO/pacpu
bash build.sh llama2_7b 1
cd ~/NEO
```

Switching between the two libraries is controlled by
`library` field in the JSON configs in `evaluation/configs/`.

---

## 4. Download model weights

Llama-3-8B is gated; you need a HuggingFace token with the Meta license
already accepted at huggingface.co/meta-llama/Llama-3.1-8B.

```bash
export HF_TOKEN=hf_<your-token>
huggingface-cli login --token $HF_TOKEN

mkdir -p ~/weights

huggingface-cli download meta-llama/Llama-3.1-8B \
    --local-dir ~/weights/Llama-3-8B \
    --exclude "*.pth"

# (Optional, for the fig6c baseline)
huggingface-cli download meta-llama/Llama-2-7b-hf \
    --local-dir ~/weights/Llama-2-7b-hf \
    --exclude "*.pth"
```

### 4.1 Confirm the config JSON points at the weights

`evaluation/configs/config-a10-8b.json` ships with
`"model_path": "/home/ubuntu/weights/Llama-3-8B"` — that should already
match. If you downloaded somewhere else, edit it:

```bash
sed -i 's|/home/ubuntu/weights|'"$HOME"'/weights|' \
    evaluation/configs/config-a10-8b.json
sed -i 's|/home/ubuntu/weights|'"$HOME"'/weights|' \
    evaluation/configs/config-t4-7b.json
cat evaluation/configs/config-a10-8b.json
```

---

## 5. Download evaluation data

### 5.1 WikiText-2 (perplexity)

```bash
mkdir -p ~/NEO/benchmarks/data
cd /tmp
curl -L https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip -o wt2.zip
unzip -q wt2.zip -d ~/NEO/benchmarks/data/
ls ~/NEO/benchmarks/data/wikitext-2-raw/wiki.test.raw
```

### 5.2 HumanEval

```bash
cd ~/NEO/benchmarks/data
curl -L https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz -o HumanEval.jsonl.gz
gunzip HumanEval.jsonl.gz
ls HumanEval.jsonl
```

### 5.3 CNN/DailyMail sample for ROUGE

```bash
python - <<'PY'
import json
from datasets import load_dataset
ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
with open("/home/ubuntu/NEO/benchmarks/data/cnndm_test_sample.jsonl", "w") as f:
    for i, row in enumerate(ds):
        if i >= 500: break
        f.write(json.dumps({"article": row["article"], "highlights": row["highlights"]}) + "\n")
print("done")
PY
ls ~/NEO/benchmarks/data/cnndm_test_sample.jsonl
```

---

## 6. Create the report directory structure

Every subsequent stage writes into `~/report`, so the final download
is one directory:

```bash
mkdir -p ~/report/{01_baseline,02_smoke,03_quality,04_perf,05_ablation}
```

---

## 7. Sanity smoke tests (cheap — run before paying for the full sweeps)

### 7.1 Component unit tests (CPU-only, ~1 minute)

```bash
cd ~/NEO
pytest tests/test_int8_transfer.py tests/test_int8_correctness.py -v \
    2>&1 | tee ~/report/02_smoke/unit_tests.log
```

All `test_quantize_*` and `test_logit_mse_fp16_vs_int8` should pass.

### 7.2 Single-prompt smoke of the int8 path

```bash
python examples/example.py \
    --model-path ~/weights/Llama-3-8B \
    --model-name llama3_8b \
    --int8-cpu-kv --quant-granularity per-token \
    2>&1 | tee ~/report/02_smoke/int8_smoke.log
```

If this crashes inside pacpu or inside `_swap_blocks_int8_cpu_kv`,
stop here and diagnose. Common failures:

- `libpacpu-llama3_8b-tp1.so: cannot open shared object` — `library`
  field in the JSON config doesn't match `pacpu/build/` output. Run
  `ls pacpu/build/libpacpu-*.so` and set `library` to whatever's there.
- `RuntimeError: expected int8 input, got torch.float16` inside
  `dequantize_int8` — M6 flag didn't route cleanly; check that
  `transformer_layer.py:336` branch was applied.
- `ValueError: --quant-granularity=per-channel is incompatible with --int8-cpu-kv` —
  this is the intentional guard from `block_swapper.py`. Drop
  per-channel or drop `--int8-cpu-kv`.

---

## 8. Baseline NEO (fp16) + vLLM reproduction (Stage 1 → fig10a.pdf)

The paper ships `evaluation/reproduce-fig10a.py` for Llama-3-8B / A10G
— that's our baseline reference. Defaults use 100 requests for a quick
demo; bump to 2000 for paper-quality numbers.

```bash
cd ~/NEO

# Bump request count if you want the full curve. Find the line that
# sets the number and increase to 2000. The script's variable name
# varies by revision.
grep -n "num_requests\|N_REQUESTS\|num_data" evaluation/reproduce-fig10a.py

# Start the sysmon in a background shell (capture peak memory).
( while true; do
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" \
        >> ~/report/01_baseline/sysmon.log
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader \
        >> ~/report/01_baseline/sysmon.log 2>&1
    free -m | head -2 >> ~/report/01_baseline/sysmon.log
    sleep 15
  done ) &
SYSMON_PID=$!

# Run the baseline.
python evaluation/reproduce-fig10a.py 2>&1 | tee ~/report/01_baseline/fig10a.log

# Stop sysmon.
kill $SYSMON_PID 2>/dev/null

# Collect artifacts.
cp evaluation/fig10a.pdf ~/report/01_baseline/
cp -r evaluation/results ~/report/01_baseline/raw 2>/dev/null || true
```

Your baseline is now in `~/report/01_baseline/`. This is the reference
curve everything else compares against.

### 8.1 (Optional) fig6c on Llama-2-7B

```bash
python evaluation/reproduce-fig6c.py 2>&1 | tee ~/report/01_baseline/fig6c.log
cp evaluation/fig6c.pdf ~/report/01_baseline/
```

---

## 9. Quality evals — the M6 acceptance gate

**Run this BEFORE performance benchmarks.** A fast int8 throughput
number on broken logits is worthless. Bars per
`docs/int8-design.md §7`:

| Metric | FP16 | INT8 bar |
| --- | --- | --- |
| Perplexity (WikiText-2) | ~5.4 (8B) | Δ < 0.05 |
| HumanEval pass@1 | ~0.32 (8B) | Δ ≤ 1 problem (≈ 0.6 %) |
| Logit MSE | 0 | < 1e-3 |

### 9.1 Perplexity (HF adapter — no server required)

```bash
cd ~/NEO
for v in fp16-neo int8-cpu-kv int8-transfer; do
    python benchmarks/run_perplexity.py \
        --variant $v \
        --quant-granularity per-token \
        --model-path ~/weights/Llama-3-8B \
        --window 2048 \
        --max-windows 200 \
        --out ~/report/03_quality 2>&1 \
        | tee ~/report/03_quality/perp_${v}.log
done
```

Check the Δ PPL between `fp16-neo` and each int8 variant:

```bash
grep "ppl=" ~/report/03_quality/perp_*.log
```

**If any int8 variant's Δ > 0.05, STOP** and diagnose before
proceeding. Possible causes:

- Hooks didn't install (see the `[info] installed N KV quant hooks`
  line in each log — N should be 64 for Llama-3-8B: 32 layers × 2).
- Wrong granularity wired — check the exact `--quant-granularity`
  string.
- Torch version mismatch between `transformers` and model weights.

### 9.2 HumanEval (through the real server — tests the engine end-to-end)

```bash
for v in fp16-neo int8-cpu-kv int8-transfer vllm; do
    python benchmarks/run_humaneval.py \
        --variant $v \
        --config evaluation/configs/config-a10-8b.json \
        --n 164 \
        --max-tokens 512 \
        --out ~/report/03_quality 2>&1 \
        | tee ~/report/03_quality/he_${v}.log
done
```

Look for `pass@1 = 0.XXX` in each log. Δ between fp16-neo and the int8
variants must be ≤ 1 problem (~0.006 absolute).

### 9.3 ROUGE

```bash
for v in fp16-neo int8-cpu-kv int8-transfer vllm; do
    python benchmarks/run_rouge.py \
        --variant $v \
        --config evaluation/configs/config-a10-8b.json \
        --n 500 \
        --max-tokens 128 \
        --out ~/report/03_quality 2>&1 \
        | tee ~/report/03_quality/rouge_${v}.log
done
```

### 9.4 Logit-MSE integration test

```bash
NEO_TEST_MODEL_PATH=~/weights/Llama-3-8B \
    pytest tests/test_int8_correctness.py::test_logit_mse_fp16_vs_int8 -v \
    2>&1 | tee ~/report/03_quality/logit_mse.log
```

Expected: printout like `[test_logit_mse_fp16_vs_int8] MSE = 4.2e-05`
followed by `PASSED`. Bar is 1e-3.

---

## 10. Performance benchmarks (Stage 6)

Only enter this stage after §9 has passed its bars.

### 10.1 Throughput

```bash
cd ~/NEO
for v in fp16-neo int8-cpu-kv int8-transfer vllm; do
    python benchmarks/run_throughput.py \
        --variant $v \
        --quant-granularity per-token \
        --config evaluation/configs/config-a10-8b.json \
        --num-requests 2000 \
        --workloads azure-code,osc,synthetic \
        --out ~/report/04_perf 2>&1 \
        | tee ~/report/04_perf/tput_${v}.log
done
```

### 10.2 Latency-vs-rate (this is how you build fig6c-style plots for int8)

```bash
for v in fp16-neo int8-cpu-kv int8-transfer vllm; do
    for rate in 0.5 1.0 2.0 4.0 8.0; do
        python benchmarks/run_latency.py \
            --variant $v \
            --quant-granularity per-token \
            --workload azure-code \
            --rate $rate \
            --num-requests 2000 \
            --config evaluation/configs/config-a10-8b.json \
            --out ~/report/04_perf 2>&1 \
            | tee -a ~/report/04_perf/lat_${v}.log
    done
done
```

### 10.3 Memory footprint snapshot

Captures peak RSS during a fixed-size throughput run — good slide
material ("int8 cuts CPU cache in half").

```bash
for v in fp16-neo int8-cpu-kv; do
    python benchmarks/run_throughput.py \
        --variant $v --quant-granularity per-token \
        --config evaluation/configs/config-a10-8b.json \
        --num-requests 500 --workloads synthetic \
        --out /tmp/discard 2>&1 &
    PID=$!
    while kill -0 $PID 2>/dev/null; do
        ps -o rss= -p $PID >> ~/report/04_perf/rss_${v}.txt
        sleep 1
    done
    wait $PID
done
echo "peak RSS (KB):"
for v in fp16-neo int8-cpu-kv; do
    peak=$(sort -n ~/report/04_perf/rss_${v}.txt | tail -1)
    echo "  $v: $peak"
done | tee ~/report/04_perf/rss_summary.txt
```

Expectation: `int8-cpu-kv` peak RSS ≈ ½ of `fp16-neo` peak RSS.

---

## 11. M10 — granularity ablation (Stage 7)

```bash
cd ~/NEO
bash scripts/run_ablation.sh \
    --variant both \
    --config evaluation/configs/config-a10-8b.json \
    --n 1000 \
    2>&1 | tee ~/report/05_ablation/ablation.log

cp -r benchmarks/results ~/report/05_ablation/raw
cp benchmarks/results/ablation_summary.csv ~/report/
```

The sweep runs every combination of
`{int8-cpu-kv, int8-transfer} × {per-token, per-channel, per-token-per-head}`,
but skips `int8-cpu-kv × per-channel` (intentional — documented in
`block_swapper.py` and `docs/int8-design.md §3`). That skip will appear
as a `[skip]` line in the log; call it out on the limitations slide.

---

## 12. Collect and download

### 12.1 Final sanity tree

```bash
tree -L 3 ~/report
```

Expected shape:

```
~/report/
├── ablation_summary.csv
├── 01_baseline/
│   ├── fig10a.pdf
│   ├── fig10a.log
│   ├── sysmon.log
│   └── raw/
├── 02_smoke/
│   ├── unit_tests.log
│   └── int8_smoke.log
├── 03_quality/
│   ├── perp_*.log + perplexity_*.csv
│   ├── he_*.log   + humaneval_*.csv
│   ├── rouge_*.log + rouge_*.csv
│   └── logit_mse.log
├── 04_perf/
│   ├── tput_*.log + throughput_*.csv
│   ├── lat_*.log  + latency_*.csv
│   └── rss_*.txt  + rss_summary.txt
└── 05_ablation/
    ├── ablation.log
    └── raw/
```

### 12.2 Tarball and download

On the instance:

```bash
cd ~
tar czf neo_report.tar.gz report/
du -h neo_report.tar.gz
```

From your laptop:

```bash
scp -i ~/.ssh/neo-runbook.pem \
    ubuntu@<PUB_IP>:~/neo_report.tar.gz \
    ./neo_report.tar.gz
tar xf neo_report.tar.gz
```

### 12.3 **Stop the instance immediately**

```bash
# from your laptop
aws ec2 stop-instances --instance-ids <INSTANCE_ID>
```

Or terminate if you don't need it again:

```bash
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
```

---

## 13. Generating the report / slides

The numbers you now have map to slide / report sections as follows:

| Slide / section | Source CSVs |
| --- | --- |
| Motivation (CPU KV footprint) | `04_perf/rss_summary.txt`, `docs/int8-design.md §1` |
| Design decisions | `docs/int8-design.md` (pre-filled) |
| Correctness (quality bar cleared) | `03_quality/perplexity_*.csv`, `humaneval_*_summary.csv`, `rouge_*_summary.csv`, `logit_mse.log` |
| Memory reduction | `04_perf/rss_summary.txt` |
| Throughput head-to-head | `04_perf/throughput_*.csv` |
| Latency curves | `04_perf/latency_*.csv` — feed to `scripts/plot_headtohead.py` if present, else matplotlib |
| Granularity ablation | `ablation_summary.csv` |
| Limitations | see §14 below |

Fill in the TODO-marked tables in `docs/results.md` using the above —
the skeleton is already in the right order for a paper/slides.

To regenerate any plots from the raw CSVs:

```bash
# From wherever you extracted neo_report.tar.gz
python ~/NEO/scripts/plot_headtohead.py \
    --throughput-dir ./report/04_perf \
    --latency-dir    ./report/04_perf \
    --out ./report/plots
```

---

## 14. Honest limitations to put on the slide

- `_swap_blocks_int8_cpu_kv` and `_swap_blocks_int8` are Python loops
  (TODO: fused `swiftllm_c.swap_blocks_int8_*` kernels). The
  `int8-cpu-kv` / `int8-transfer` throughput numbers are a lower bound.
  See [swiftllm/worker/block_swapper.py:189](swiftllm/worker/block_swapper.py:189).
- Perplexity uses an HF Transformers adapter with fake-quantization
  hooks, not the full NEO engine — a 2048-token window never triggers
  offload so going through the engine would be vacuous. The quantizer
  is the same (`swiftllm.worker.quantize.roundtrip_int8`); what is not
  tested through this path is the fused-dequant ISPC/C++ kernel. That
  is covered separately by HumanEval + ROUGE (which do go through the
  engine and do see offload). See
  [docs/results.md §6](docs/results.md) and
  [benchmarks/_quant_hooks.py](benchmarks/_quant_hooks.py).
- `int8-cpu-kv × per-channel` is unsupported by design (online append
  path cannot re-quantize a whole block per token). The ablation sweep
  skips this cell. See [docs/int8-design.md §3](docs/int8-design.md).
- GPU KV cache stays FP16 — explicit M6 constraint. This run does not
  exercise any GPU-side quantization.
- Numbers are on A10G 24 GB, not H100. An H100 would reduce absolute
  latencies but not change the int8-vs-fp16 *ratios*, which is what
  the slide story is about.

---

## 15. Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ispc: command not found` after §2.3 | PATH not refreshed | `hash -r && which ispc` |
| `g++: unrecognized argument '-std=c++20'` during pacpu build | wrong g++ pinned | edit `pacpu/build.sh` to force `g++-13` |
| `torch.cuda.OutOfMemoryError` on baseline | `gpu_memory_utilization` too high | lower to 0.95 in `config-a10-8b.json` |
| `Error 429 / gated repo` on HF download | license not accepted | visit `huggingface.co/meta-llama/Llama-3.1-8B`, accept |
| Perplexity adapter prints "0 KV quant hooks" | `transformers` version too old, no `k_proj` name | upgrade `transformers` to 4.44+ |
| HumanEval server fails to start with `--int8-cpu-kv` | flag not wired in `evaluation/server.py` | apply the M6 flag passthrough from `benchmarks/_common.py:resolve_variant` |
| `ablation_summary.csv` is empty | regex in `scripts/run_ablation.sh` bottom-of-script Python block didn't match filenames | adjust the regex to match the actual CSV naming from your runs |
| Instance disconnects mid-run | SSH timeout | always run inside `tmux` — see §16 |

---

## 16. Running under tmux (strongly recommended)

Every `python ...` command in §8-§11 should run inside a tmux session so
an SSH drop doesn't kill your multi-hour sweep:

```bash
sudo apt-get install -y tmux
tmux new -s neo
# ... run commands ...
# detach:    Ctrl-b, then d
# reattach:  tmux attach -t neo
```

---

## 17. Cost accounting (fill in as you go)

Track runtime at each stage so the final slide can show total compute
cost. Save to `~/report/cost.txt`:

```bash
echo "stage,start_utc,end_utc" > ~/report/cost.txt
# Before each stage:
echo "stage_N,$(date -u +%Y-%m-%dT%H:%M:%SZ),...,0" >> ~/report/cost.txt
```

Or just check the AWS console's billing dashboard at the end — usually
accurate to the cent.

---

*End of runbook.*
