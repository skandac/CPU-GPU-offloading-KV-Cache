"""
benchmarks/run_humaneval.py

HumanEval pass@1 harness. Generates one completion per problem through
the running server (OpenAI-compatible /v1/completions endpoint), then
executes each completion in a subprocess against the problem's test
suite and counts passes.

USAGE:
    # (1) start the server separately — or rely on --variant to start one
    python benchmarks/run_humaneval.py \\
        --variant int8-transfer \\
        --config evaluation/configs/config-a10-8b.json \\
        --n 164 \\
        --max-tokens 512

OUTPUT:
    benchmarks/results/humaneval_{variant}.csv
        columns: variant, task_id, passed, wall_s, completion_len
    benchmarks/results/humaneval_{variant}_summary.csv
        columns: variant, n, passed, pass_at_1

DATA:
    Expects HumanEval JSONL at:
        benchmarks/data/HumanEval.jsonl
    (https://github.com/openai/human-eval/blob/master/data/HumanEval.jsonl.gz)

SAFETY:
    HumanEval runs arbitrary model-generated Python. We sandbox each test
    in a separate Python process with a short wall clock timeout. This is
    NOT a real sandbox — do not run outside a throwaway VM. The upstream
    human_eval library ships a `execution.check_correctness` with the same
    caveat.

    TODO(gpu-verify): if `human_eval` is installed in the env, prefer its
    `check_correctness` over the local subprocess runner below — it has
    the same safety posture but better-tested timeout handling.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Optional

from _common import (
    resolve_variant,
    ensure_out_dir,
    VARIANTS,
    EVAL_DIR,
    start_server, stop_server,
)

# api_client is under evaluation/; _common already puts it on sys.path.
from api_client import request_completions  # noqa: E402


HUMANEVAL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "HumanEval.jsonl"
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_problems(path: str = HUMANEVAL_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(
            f"[err] {path} not found.\n"
            f"      Grab HumanEval.jsonl.gz from openai/human-eval and gunzip into\n"
            f"      {os.path.dirname(path)}/.\n"
        )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Execution sandbox
# ---------------------------------------------------------------------------
_RUNNER_TEMPLATE = textwrap.dedent('''\
    import sys, json, traceback
    problem = json.loads(sys.stdin.read())
    completion = problem["completion"]
    prompt     = problem["prompt"]
    test       = problem["test"]
    entry      = problem["entry_point"]

    program = prompt + completion + "\\n" + test + f"\\ncheck({entry})\\n"
    try:
        exec_globals = {{}}
        exec(program, exec_globals)
        print("PASS")
    except Exception:
        traceback.print_exc()
        print("FAIL")
''')


def check_correctness(problem: dict, completion: str, timeout_s: float = 10.0) -> bool:
    """Execute problem['test'] against `completion`. Returns True iff "PASS"."""
    payload = {
        "prompt":      problem["prompt"],
        "completion":  completion,
        "test":        problem["test"],
        "entry_point": problem["entry_point"],
    }
    try:
        out = subprocess.run(
            [sys.executable, "-c", _RUNNER_TEMPLATE],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False
    return "PASS" in out.stdout.splitlines()[-1:] if out.stdout else False


# ---------------------------------------------------------------------------
# Generation + scoring
# ---------------------------------------------------------------------------
async def generate_one(api_url: str, prompt: str, model_path: str, max_tokens: int) -> str:
    """
    Request a single completion. api_client.request_completions returns
    the raw string completion (the part after the prompt).
    """
    return await request_completions(api_url, prompt, max_tokens, model_path)


def trim_to_first_function_end(completion: str) -> str:
    """HumanEval best practice: stop at the first dedent / 'def ' / '\\nclass '.
    Keeps the model from continuing past the target function into garbage."""
    stops = ["\nclass ", "\ndef ", "\nif __name__", "\nprint(", "\n#"]
    cut = len(completion)
    for s in stops:
        i = completion.find(s)
        if i != -1:
            cut = min(cut, i)
    return completion[:cut]


async def main_async():
    args = parse_args()
    variant_spec = resolve_variant(args.variant)
    print(f"[info] variant: {args.variant} — {variant_spec.description}")

    with open(args.config) as f:
        config = json.load(f)
    model_path = config["model_path"]

    problems = load_problems()[: args.n]

    start_server(variant_spec.server_name, config)
    api_url = "http://localhost:8000/v1/completions"
    rows = []
    try:
        for idx, prob in enumerate(problems):
            t0 = time.time()
            raw = await generate_one(api_url, prob["prompt"], model_path, args.max_tokens)
            completion = trim_to_first_function_end(raw)
            passed = check_correctness(prob, completion, timeout_s=args.exec_timeout)
            rows.append({
                "variant": args.variant,
                "task_id": prob["task_id"],
                "passed": int(passed),
                "wall_s": round(time.time() - t0, 3),
                "completion_len": len(completion),
            })
            if (idx + 1) % 10 == 0:
                acc = sum(r["passed"] for r in rows) / len(rows)
                print(f"  [{idx+1}/{len(problems)}] running pass@1 = {acc:.3f}")
    finally:
        stop_server()

    out_dir = ensure_out_dir(args.out)
    per_path = os.path.join(out_dir, f"humaneval_{args.variant}.csv")
    with open(per_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {per_path}")

    n_pass = sum(r["passed"] for r in rows)
    pass_at_1 = n_pass / max(len(rows), 1)
    sum_path = os.path.join(out_dir, f"humaneval_{args.variant}_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n", "passed", "pass_at_1"])
        w.writerow([args.variant, len(rows), n_pass, pass_at_1])
    print(f"[ok] wrote {sum_path}")
    print(f"[result] {args.variant}: pass@1 = {pass_at_1:.4f} on n={len(rows)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HumanEval pass@1 harness (M9)")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--config", default=os.path.join(EVAL_DIR, "configs", "config-a10-8b.json"))
    p.add_argument("--n", type=int, default=164, help="Number of problems (full set = 164)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--exec-timeout", type=float, default=10.0,
                   help="Per-problem subprocess timeout in seconds")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async())
