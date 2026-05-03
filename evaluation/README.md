# Evaluation

Paper-faithful reproduction harness plus the supporting infrastructure
(server, benchmark driver, illustrator) used by both the original NEO
reproductions and the INT8 evaluations in this fork.

## Layout

```
evaluation/
├── reproduce-fig6c.py        Paper Fig. 6c — load-latency on T4 + Llama-2-7B + OSC
├── reproduce-fig10a.py       Paper Fig. 6b/10a — A10G + Llama-3.1-8B + AC
├── server.py                 Server lifecycle (NEO + vLLM start/stop) used by the
│                             reproduction scripts and benchmarks/
├── benchmark.py              Request driver (workload load, async send, timing)
├── api_client.py             OpenAI-style HTTP client used to issue requests
├── illustrator.py            Plot helpers (load-latency, throughput curves)
├── configs/                  JSON engine configs
│   ├── config-t4-7b.json     T4 / Llama-2-7B (paper Fig. 6c hardware)
│   └── config-a10-8b.json    A10G / Llama-3.1-8B (paper Fig. 6b hardware)
└── data/                     Workload prompt-length specs (JSON)
```

## Reproducing the paper

The two flagship NEO figures reproduce with one command each.

### Fig. 6c — T4 + Llama-2-7B + OpenAI Summarization

```bash
python evaluation/reproduce-fig6c.py
# output: evaluation/fig6c.pdf
```

### Fig. 6b / Fig. 10a — A10G + Llama-3.1-8B + Azure Code

```bash
python evaluation/reproduce-fig10a.py
# output: evaluation/fig10a.pdf (and fig6b.pdf depending on axis)
```

Both scripts manage their own server lifecycle on TCP port 8000. Make
sure the corresponding model weights and pacpu library are built (see
top-level `README.md` and `ec2_run_instructions.md`).

## How this connects to the INT8 work

The reproduction scripts above measure NEO's FP16 baseline only — they
are paper-faithful and intentionally untouched by this fork. The
INT8 evaluations live in a sibling directory:

| Question | Where to look |
|---|---|
| Does INT8 transfer reduce latency? | `benchmarks/run_latency.py` |
| Does INT8 transfer preserve perplexity? | `benchmarks/run_perplexity.py` |
| Which granularity is best? | `scripts/run_ablation.sh` |
| Head-to-head plots (FP16 vs. INT8)? | `scripts/plot_headtohead.py` |

The infrastructure in this folder (`server.py`, `benchmark.py`,
`api_client.py`, `configs/`) is shared between the paper reproduction
and the INT8 work — the INT8 benchmarks reuse the same server-start
helpers and config files, just with different engine flags.

## Configs

Both JSON configs in `configs/` ship with `model_path` placeholders
pointing at `/home/ubuntu/weights/...` — the layout used by the AWS
runbook. Edit `model_path` to match your local weights directory before
running anything.

The INT8 paths read the same configs and respect the same
`block_size`, `max_num_seqs`, `swap_space`, and `library` fields. INT8
is selected via engine-config flags (`int8_transfer`, `int8_cpu_kv`)
or CLI flags on `benchmarks/run_*.py`, not via a separate config file.

## Cost notes (AWS reproduction)

A full reproduction of Fig. 6c + Fig. 6b on `g5.16xlarge` takes
~8 hours of wall-clock and costs ~$120 of on-demand EC2. See
`ec2_run_instructions.md` for the step-by-step runbook.
