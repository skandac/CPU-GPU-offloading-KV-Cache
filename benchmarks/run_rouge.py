"""
benchmarks/run_rouge.py

ROUGE-1 / ROUGE-2 / ROUGE-L on a CNN/DailyMail sample. For each article
in the sample we generate a summary through the NEO-or-vLLM server and
score it against the reference highlights.

USAGE:
    python benchmarks/run_rouge.py \\
        --variant int8-transfer \\
        --config evaluation/configs/config-a10-8b.json \\
        --n 500 \\
        --max-tokens 128

OUTPUT:
    benchmarks/results/rouge_{variant}.csv          (per-article scores)
    benchmarks/results/rouge_{variant}_summary.csv  (mean r1 / r2 / rL)

DATA:
    Expects CNN/DailyMail sample at benchmarks/data/cnndm_test_sample.jsonl
    — one JSON object per line with fields {"article": str, "highlights": str}.
    The full test set is 11490 examples; 500 is the conventional
    evaluation subset used in summarization leaderboards.

DEPENDENCIES:
    - rouge_score    (pip install rouge-score)   official Google impl
    - datasets       (optional — to stream CNN/DailyMail directly)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
from typing import Optional

from _common import (
    resolve_variant,
    ensure_out_dir,
    VARIANTS,
    EVAL_DIR,
    start_server, stop_server,
)
from api_client import request_completions  # noqa: E402


CNNDM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "cnndm_test_sample.jsonl"
)


def load_sample(path: str = CNNDM_PATH, n: int = 500) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(
            f"[err] {path} not found.\n"
            f"      Build a sample with:\n"
            f"        python -c \"from datasets import load_dataset; "
            f"ds = load_dataset('cnn_dailymail','3.0.0',split='test'); "
            f"ds.select(range(500)).to_json('{path}')\"\n"
        )
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= n:
                    break
    return rows


# ---------------------------------------------------------------------------
# Scoring — thin wrapper around google-research/rouge_score
# ---------------------------------------------------------------------------
def build_scorer():
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise SystemExit(
            "rouge_score is not installed. Run `pip install rouge-score`."
        ) from e
    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def score_one(scorer, prediction: str, reference: str) -> dict:
    s = scorer.score(reference, prediction)
    return {
        "rouge1_f": s["rouge1"].fmeasure,
        "rouge2_f": s["rouge2"].fmeasure,
        "rougeL_f": s["rougeL"].fmeasure,
    }


# ---------------------------------------------------------------------------
# Prompt template — matches the OSC workload's zero-shot style
# ---------------------------------------------------------------------------
_SUMMARY_PROMPT = (
    "Article:\n{article}\n\n"
    "Write a 3-sentence summary of the above article. Do not add any preamble.\n\n"
    "Summary:"
)


async def generate_summary(api_url: str, article: str, model_path: str, max_tokens: int) -> str:
    prompt = _SUMMARY_PROMPT.format(article=article)
    return await request_completions(api_url, prompt, max_tokens, model_path)


def trim_summary(text: str) -> str:
    """Cut at the first double newline — models love to keep going."""
    i = text.find("\n\n")
    return text[:i].strip() if i != -1 else text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main_async():
    args = parse_args()
    variant_spec = resolve_variant(args.variant)
    print(f"[info] variant: {args.variant} — {variant_spec.description}")

    with open(args.config) as f:
        config = json.load(f)
    model_path = config["model_path"]

    sample = load_sample(n=args.n)
    scorer = build_scorer()

    start_server(variant_spec.server_name, config)
    api_url = "http://localhost:8000/v1/completions"
    rows = []
    try:
        for i, ex in enumerate(sample):
            t0 = time.time()
            raw = await generate_summary(api_url, ex["article"], model_path, args.max_tokens)
            pred = trim_summary(raw)
            s = score_one(scorer, pred, ex["highlights"])
            rows.append({
                "variant":    args.variant,
                "idx":        i,
                "wall_s":     round(time.time() - t0, 3),
                "pred_len":   len(pred),
                "ref_len":    len(ex["highlights"]),
                **s,
            })
            if (i + 1) % 25 == 0:
                r1 = sum(r["rouge1_f"] for r in rows) / len(rows)
                print(f"  [{i+1}/{len(sample)}] running ROUGE-1 = {r1:.4f}")
    finally:
        stop_server()

    out_dir = ensure_out_dir(args.out)
    per_path = os.path.join(out_dir, f"rouge_{args.variant}.csv")
    with open(per_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {per_path}")

    mean = lambda k: sum(r[k] for r in rows) / max(len(rows), 1)  # noqa: E731
    sum_path = os.path.join(out_dir, f"rouge_{args.variant}_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n", "rouge1_f", "rouge2_f", "rougeL_f"])
        w.writerow([args.variant, len(rows),
                    mean("rouge1_f"), mean("rouge2_f"), mean("rougeL_f")])
    print(f"[ok] wrote {sum_path}")
    print(f"[result] {args.variant}: R1={mean('rouge1_f'):.4f} "
          f"R2={mean('rouge2_f'):.4f} RL={mean('rougeL_f'):.4f} on n={len(rows)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CNN/DailyMail ROUGE harness (M9)")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--config", default=os.path.join(EVAL_DIR, "configs", "config-a10-8b.json"))
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async())
