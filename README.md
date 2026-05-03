# NEO + INT8 KV-Cache Quantization

This repository extends [NEO](https://yangzhou1997.github.io/paper/neo_mlsys25.pdf) — a CPU/GPU offloading inference engine — with an **INT8 quantized KV-cache transfer path** that halves the PCIe bandwidth required by NEO's swap operations while preserving model quality.

## Background — NEO

Online LLM inference powers many exciting applications such as intelligent chatbots and autonomous agents. Modern LLM inference engines widely rely on request batching to improve inference throughput, aiming to make it cost-efficient when running on expensive GPU accelerators. However, the limited GPU memory has largely limited the batch size achieved in practice, leaving significant GPU compute resources wasted.

NEO is an online LLM inference system that offloads part of attention compute and KV cache states from the GPU to the local host CPU, effectively increasing the GPU batch size and thus inference throughput. To this end, NEO proposes asymmetric GPU-CPU pipelining and load-aware scheduling to balance GPU and CPU loads and fully utilize their compute and memory resources. The original NEO paper (MLSys'25) is [here](https://yangzhou1997.github.io/paper/neo_mlsys25.pdf).

## What this fork adds — INT8 KV-cache transfer (M5–M11)

NEO's design moves blocks of KV cache between CPU and GPU memory across PCIe whenever a request migrates between residence tiers. By default each block is transferred in FP16. This fork adds an INT8 transfer path that compresses each block to 8 bits with symmetric per-token scaling before it crosses PCIe and decompresses on the receiving side — halving the bytes on the wire while leaving the on-device cache layout in FP16.

Highlights:

- **`swiftllm/worker/quantize.py`** — pure-PyTorch symmetric INT8 quantizer with three granularity options (per-token, per-channel, per-token-per-head).
- **`--int8-transfer`** and **`--int8-cpu-kv`** CLI flags exposed by the engine entry point and the example script.
- **`benchmarks/`** — runnable latency, throughput, perplexity, ROUGE, and HumanEval benchmarks comparing FP16 vs. INT8 variants.
- **`scripts/run_ablation.sh`** — granularity ablation harness (M10).
- **`tests/test_int8_*.py`** — 10/10 passing unit tests covering quantizer round-trip bounds, dtype contracts, and broadcasting.
- **`docs/int8-design.md`** — design doc for the quantization path.
- **`ec2_run_instructions.md`** — end-to-end runbook from a clean AWS account through baseline reproduction and INT8 evaluation.

### INT8 quick-results summary

Measured on Llama-3-8B / NVIDIA A100, synthetic workload, 30 requests per rate point:

| Metric (p95) | FP16 | INT8 transfer | Δ |
|---|---|---|---|
| Per-token latency at 0.5 req/s | 34.7 ms | 18.1 ms | **−48%** |
| TTFT at 0.5 req/s | 2.91 s | 0.16 s | **−94%** |
| End-to-end at 0.5 req/s | 6.95 s | 3.78 s | **−46%** |
| WikiText-2 perplexity (100 windows × 2048 tokens) | 6.3312 | 6.3322 | **+0.015%** |

INT8 transfer wins decisively at low-to-moderate load (where the swap path is on the critical path) and matches FP16 at saturated load — with no measurable model-quality regression.

## Requirements

Python >= 3.10
PyTorch >= 2.4

2 versions of g++ (see `pacpu/build.sh` for more details):

- one >= 13 (for compiling CPU kernel)
- the other < 13 (for passing the NVCC version check)

Intel ISPC compiler == 1.23, which can be installed by `sudo snap install ispc --channel latest/edge`

## Installation

1. Clone the NEO repository and `cd` into the repo.

2. Install dependencies by `pip install -r requirements.txt.`

3. Install the swiftLLM library to your local environment by `pip install -e .`

4. Build and install auxiliary GPU operators library by `pip install -e csrc`

5. Build the CPU operator library by 

   ```bash
   cd pacpu
   bash build.sh <model-name> <tensor-parallel-degree> 
   # e.g bash build.sh llama2_7b 1
   cd ..
   ```

## Offline Example

```bash
cd NEO
python examples/example.py --model-path ... --model-name ...
# e.g. python examples/example.py --model-path /home/ubuntu/weights/Llama-2-7b-hf/ --model-name llama2_7b
```

Run `python examples/example.py --help` to see more options.

## Performance Results

### Load-latency Curves

The figure below (Figure 6c in the paper) shows online latencies of NEO and other baselines under different request rates.

vLLM-256 and vLLM-512 designate vLLM with chunked-prefilling at the chunk size of 256 and 512 tokens, respectively.

![image-20250221101244560](docs/load-latency.png)

- Hardware: AWS g4.4xlarge instance, with Tesla T4 GPU, 8 cores of Xeon P-8259CL CPU, and 64 GB main memory.
- Model: LLaMa-2-7B
- Workload: OpenAI summarization comparison ([CarperAI](https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons.))

### Generation Throughput

The figure below (Figure 10a in the paper) shows NEO's throughput gains over the non-CPU-offloading baseline under different workloads. NEO achieves up to 12.2%, 13.3%, 29.7%, and 79.3% higher throughput over the baseline under different CPU capacities.

![image-20250221101309717](docs/cpu-sensitivity.png)

- Hardware: AWS g5.nxlarge instances (n=2,4,8,16), with A10 GPU, 2n cores of EPYC 7R32 CPU, and 16n GB main memory.
- Model: LLaMa-3-8B
- Workload: Synthetic workloads with various input and output lengths. For a pair of input length $l_i$ and output length $l_o$, we synthesize requests with input and output lengths sampled independently and uniformly from $[0.9l_i, 1.1l_i]$ and $[0.9l_o, 1.1l_o]$, respectively. Here we fix $l_i=1000$ and pick $l_o$ from $\{50, 100, 200, 300, 400\}$.

## Reproduction

Below are instructions for reproducing Figure 6c in the paper. Instructions for Figure 10a are the same except for specific details noted in parentheses.

### With an AWS Account

1. Launch a g4dn.4xlarge (g5.16xlarge) instance in us-east-1 region with community AMI neo-ae-g4-image (neo-ae-g5-image).
2. SSH to the instance and run `mamba activate neo` in the shell.
3. run `cd NEO`
4. run `python evaluation/reproduce-fig6c.py`(`python evaluation/reproduce-fig10a.py`)

> NOTE: Although the model weights are pre-packaged in the images, the first time loading them would take about 1 hour. Therefore, it is recommended to download the weights from the internet and replace those embedded in  the image, which usually takes less than 10 min. The following script can be used to retrieve the weights from Huggingface:
>
> ```bash
> cd ~
> rm -r weights/*
> ip install 'huggingface_hub[cli]' 
> huggingface-cli login --token <your huggingface token>
> # For g5 instance:
> huggingface-cli download meta-llama/Llama-3.1-8B --local-dir weights/Llama-3-8B --exclude "*.pth"
> # For g4 instance:
> huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir weights/Llama-2-7b-hf --exclude "*.pth"
> ```
>
> Alternatively, you may use the pre-packaged weights within the image. It is possible to encounter timeout issues during the initial execution of the evaluation script due to prolonged loading times. If this occurs, simply rerunning the script should resolve the issue.

### Without an AWS Account

1. Prepare a machine with 
   - Nvidia Tesla T4 (A10G) GPU;
   - CPU with AVX2 support;
   - At least 30GB (120GB) main memory for CPU KV Cache.
   - Ubuntu >= 22.04
2. Follow the steps in the Installation section to install dependencies.
3. Download LLaMa-2-7B (LLaMa-3-8B) model weights. You can refer to the NOTE above for weight retrieving scripts.
4. Modify `model_path` entry in `evaluation/configs/config-t4-7b.json` ( `evaluation/configs/config-a10-8b.json`) to the actual path to the model weights.
5. run `python evaluation/reproduce-fig6c.py`(`python evaluation/reproduce-fig10a.py`) in top level directory of the NEO repository.

### Expected Results

- The reproduced figure fig6c.pdf (fig10a.pdf) will be produced in `evaluation` directory.
- For Figure 6c, there will be only 2 lines (Neo and vLLM). By default the script only uses a small subset (100 requests) of the original input data (2000 requests) used in the original experiment. This is for the purpose of demonstration and quick verification of the results for faster evaluation. As a result, the latency would be lower than the original figure due to less average queuing latency.
- For Figure 10a, only 2 lines (x16large and baseline) in the original figure will be drawn.

> NOTE: You can change the hyperparameters of the experiments by modifying the corresponding scripts. Please refer to comments in the code for detailed instructions.

## Reproducing the INT8 results

The INT8 evaluation suite is independent of the NEO paper reproduction
above — it runs against the same engine but with the INT8 flags
enabled. To reproduce the headline numbers in this README:

```bash
# 1. Sanity: 10 unit tests for the quantizer (CPU-only, ~10s)
pytest tests/test_int8_transfer.py -v

# 2. Smoke test: single-prompt INT8 inference end-to-end
python examples/example_int8.py \
    --model-path /path/to/Llama-3-8B \
    --model-name llama3_8b \
    --int8-cpu-kv --quant-granularity per-token

# 3. Latency: FP16 vs INT8-transfer sweep across rates
for variant in fp16-neo int8-transfer; do
  for rate in 0.5 1.0 2.0; do
    python benchmarks/run_latency.py \
        --variant $variant --workload synthetic --rate $rate \
        --num-requests 30 \
        --config evaluation/configs/config-a10-8b.json
  done
done

# 4. Perplexity: WikiText-2 quality check
python benchmarks/run_perplexity.py --variant fp16-neo \
    --model-path /path/to/Llama-3-8B
python benchmarks/run_perplexity.py --variant int8-transfer \
    --quant-granularity per-token \
    --model-path /path/to/Llama-3-8B

# 5. Granularity ablation (M10)
bash scripts/run_ablation.sh --variant int8-transfer \
    --config evaluation/configs/config-a10-8b.json
```

Outputs land under `benchmarks/results/` and `docs/figures/`. The full
runbook from a clean AWS account is in `ec2_run_instructions.md`.

For the design rationale and per-component documentation, see
`docs/int8-design.md`.

## Repository structure

```
.
├── swiftllm/                 NEO engine (Python)
│   └── worker/
│       ├── quantize.py       INT8 quantizer (this fork's contribution)
│       └── block_swapper.py  Swap path with INT8 hooks
├── csrc/                     CUDA/C++ extensions (NEO)
├── pacpu/                    CPU attention kernel (NEO; INT8 dequant added)
├── examples/                 Offline inference demos (FP16 + INT8)
├── benchmarks/               Latency / throughput / perplexity / ROUGE / HumanEval
├── evaluation/               Paper-figure reproduction harness (NEO)
├── tests/                    Unit + correctness tests for INT8 path
├── scripts/                  Run/plot/ablation helpers
├── docs/
│   ├── int8-design.md        INT8 design doc
│   └── results.md            Combined results table
└── ec2_run_instructions.md   End-to-end AWS reproduction runbook
```

## Attribution

The base NEO engine (asymmetric pipelining, paged attention, pacpu, and
the paper-reproduction scripts) is the work of the original NEO authors
(Liu et al., MLSys'25). This fork adds the INT8 KV-cache transfer path
(M5–M11) on top: the new files are listed in the "What this fork adds"
section above, and the modifications to existing NEO files are visible
via `git log` / `git blame`.
