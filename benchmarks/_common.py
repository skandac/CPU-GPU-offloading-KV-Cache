"""
benchmarks/_common.py

Shared plumbing for the M8 benchmark harness. Centralizes:
    - variant → server launch config (fp16-neo, int8-cpu-kv, int8-transfer, vllm)
    - workload loading (reuses evaluation/benchmark.py helpers)
    - CSV writing

Every M8 script (run_throughput / run_latency / run_perplexity) imports from
here so the --variant flag behaves identically across them.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

# Make evaluation/ importable — same trick scripts/run_vllm_baseline.py uses.
_THIS = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(_THIS)
EVAL_DIR = os.path.join(REPO_DIR, "evaluation")
sys.path.insert(0, EVAL_DIR)

# pylint: disable=wrong-import-position,import-error
from server import start_server, stop_server              # noqa: E402
from benchmark import run_test, prepare_real_test, prepare_mock_test  # noqa: E402


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
VARIANTS = ("fp16-neo", "int8-cpu-kv", "int8-transfer", "vllm")


@dataclass
class VariantSpec:
    """How to launch the server for a given variant."""
    server_name: str              # passed to start_server — 'ours'/'vllm'/'base'
    extra_server_args: List[str]  # extra CLI args appended to the server cmdline
    description: str


GRANULARITIES = ("per-token", "per-channel", "per-token-per-head")


def resolve_variant(variant: str, granularity: str = "per-token") -> VariantSpec:
    """
    Map a --variant string to a VariantSpec. server_name is consumed by
    evaluation/server.py:start_server — that function switches on the prefix
    ('vllm'/'ours'/'base') to pick the backend.

    The ``granularity`` argument is only consulted for INT8 variants (M10
    ablation). It is appended as ``--quant-granularity <g>`` to the server
    argv. For non-INT8 variants it is silently ignored.

    TODO(gpu-verify): int8-cpu-kv requires M6's patches to be merged. When
    they are, the extra_server_args below must match M6's CLI spelling —
    this is currently a best guess.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(f"unknown granularity: {granularity!r} (choose from {GRANULARITIES})")

    if variant == "fp16-neo":
        return VariantSpec("ours", [], "NEO with default FP16 KV cache")
    if variant == "int8-cpu-kv":
        # M6 flag name is unverified — patch here when M6 lands.
        return VariantSpec(
            "ours", ["--int8-cpu-kv", "--quant-granularity", granularity],
            f"NEO with INT8 CPU KV cache (M6), granularity={granularity}",
        )
    if variant == "int8-transfer":
        return VariantSpec(
            "ours", ["--int8-transfer", "--quant-granularity", granularity],
            f"NEO with INT8 H2D/D2H transfer (M7), granularity={granularity}",
        )
    if variant == "vllm":
        return VariantSpec("vllm", [], "vLLM baseline")
    raise ValueError(f"unknown variant: {variant!r} (choose from {VARIANTS})")


# ---------------------------------------------------------------------------
# Workload loading
# ---------------------------------------------------------------------------
WORKLOADS = ("azure-code", "osc", "synthetic")


def load_workload(
    workload: str,
    server_name: str,
    config: dict,
    nreqs: int,
):
    """
    Return (prompts, output_lens, res_prefix, model_path) for the given
    workload. Mirrors prepare_real_test / prepare_mock_test contracts so
    the result is directly usable by benchmark.py:run_test.
    """
    if workload == "synthetic":
        # Moderate prompt, moderate output — matches the defaults the paper's
        # fig10a uses as the 'base' synthetic workload.
        return prepare_mock_test(nreqs, input_len=1000, output_len=200,
                                 server_name=server_name, config=config)
    if workload in ("azure-code", "osc"):
        return prepare_real_test(workload, config, server_name)
    raise ValueError(f"unknown workload: {workload!r} (choose from {WORKLOADS})")


# ---------------------------------------------------------------------------
# Percentile + CSV helpers — kept dependency-free to avoid numpy in benchmark
# entrypoints (numpy is already a transitive dep but we don't need it here).
# ---------------------------------------------------------------------------
def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    k = (len(vs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vs) - 1)
    return vs[f] if f == c else vs[f] + (vs[c] - vs[f]) * (k - f)


def ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
