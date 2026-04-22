"""
benchmarks/run_perplexity.py

WikiText-2 perplexity — sanity check that quantization variants don't
regress generation quality. Unlike run_throughput / run_latency, this
does NOT go through the serving path; it runs the model forward pass
directly on fixed-length token windows and computes NLL.

USAGE:
    python benchmarks/run_perplexity.py \\
        --variant int8-transfer \\
        --model-path /models/Llama-2-7B \\
        --window 2048 \\
        --max-windows 200

OUTPUT:
    benchmarks/results/perplexity_{variant}.csv
        columns: variant, window_size, num_windows, nll_mean, perplexity

DATA:
    Expects WikiText-2 raw test at:
        benchmarks/data/wikitext-2-raw/wiki.test.raw
    If missing, prints instructions and exits.

VARIANTS:
    For fp16-neo / int8-cpu-kv / int8-transfer we call the NEO engine's
    forward helper. For vllm we load the model via HuggingFace transformers
    (vLLM does not expose a logits-returning forward suitable for PPL).
    This keeps the reference point honest: each variant's forward is
    evaluated on its own terms. The quantization variants apply their
    modifications to the NEO forward path and we measure the resulting
    perplexity delta.

CAVEAT:
    NEO's int8 KV cache only materializes on *offloaded* blocks. A
    2048-token perplexity window never triggers offload, so running
    perplexity through the real NEO engine with --int8-cpu-kv would
    return the FP16 number. Instead, for int8 variants we load the
    reference FP16 model via HF Transformers and register forward
    hooks that round-trip K / V through the *same* quantizer NEO uses
    (swiftllm.worker.quantize.roundtrip_int8). This isolates the
    accuracy question ("does the quantization math hurt logits") from
    the engine-implementation question ("does pacpu's fused dequant
    match spec"), which is covered separately by
    tests/test_int8_correctness.py's component arm and by the end-to-
    end HumanEval / ROUGE harnesses that DO go through the engine.

    See benchmarks/_quant_hooks.py for the hook math and a full
    list of the caveats this simulation carries.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Iterable

# Torch is a transitive dep of the NEO engine; importing at module scope
# keeps the error message useful if the env is wrong.
import torch

from _common import (
    resolve_variant,
    ensure_out_dir,
    VARIANTS, GRANULARITIES,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
WIKITEXT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "wikitext-2-raw", "wiki.test.raw"
)


def load_wikitext_tokens(tokenizer, path: str = WIKITEXT_PATH) -> torch.Tensor:
    if not os.path.exists(path):
        raise SystemExit(
            f"[err] {path} not found.\n"
            f"      Download via:\n"
            f"        mkdir -p {os.path.dirname(path)}\n"
            f"        curl -L https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip -o /tmp/wt2.zip\n"
            f"        unzip /tmp/wt2.zip -d {os.path.dirname(os.path.dirname(path))}\n"
        )
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # HF tokenizer API — handles BPE for Llama-family models.
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return ids


# ---------------------------------------------------------------------------
# Forward adapters — one per variant family
# ---------------------------------------------------------------------------
class HFForwardAdapter:
    """
    Plain HF Transformers forward — used as the reference (FP16) path
    and by the vLLM variant (vLLM doesn't expose a PPL-suitable logit
    interface, so we use HF as a clean-room reference there too).
    """

    def __init__(self, model_path: str, dtype=torch.float16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="cuda"
        ).eval()
        # head_dim is stashed for subclasses that need to reshape to
        # NEO's block layout.
        self.head_dim = getattr(
            self.model.config, "head_dim",
            self.model.config.hidden_size // self.model.config.num_attention_heads,
        )

    @torch.inference_mode()
    def logits(self, window_ids: torch.Tensor) -> torch.Tensor:
        return self.model(window_ids.unsqueeze(0).cuda()).logits[0]


class NEOForwardAdapter(HFForwardAdapter):
    """
    HF forward with a KV fake-quantization hook layered on.

    For ``variant="fp16-neo"`` the hook is not installed — the FP16
    reference and NEO's FP16 path are the same numerically (modulo
    kernel choice, which does not change the quantization question).

    For ``variant="int8-cpu-kv"`` and ``variant="int8-transfer"`` the
    hook rewrites every Llama attention K / V projection output to its
    round-tripped int8 approximation using the requested granularity.
    See benchmarks/_quant_hooks.py for the math and caveats.
    """

    def __init__(
        self,
        model_path: str,
        variant: str,
        granularity: str = "per-token",
    ):
        super().__init__(model_path)
        self.variant = variant
        self.granularity = granularity
        self._hook_handles = []
        if variant in ("int8-cpu-kv", "int8-transfer"):
            from _quant_hooks import install_kv_quant_hooks  # local import
            from swiftllm.worker.quantize import QuantGranularity
            self._hook_handles = install_kv_quant_hooks(
                self.model,
                head_dim=self.head_dim,
                granularity=QuantGranularity(granularity),
            )
            print(
                f"[info] installed {len(self._hook_handles)} KV quant hooks "
                f"(variant={variant}, granularity={granularity})"
            )

    def close(self) -> None:
        from _quant_hooks import remove_kv_quant_hooks
        remove_kv_quant_hooks(self._hook_handles)
        self._hook_handles = []


def build_adapter(variant: str, model_path: str, granularity: str = "per-token"):
    if variant == "vllm":
        return HFForwardAdapter(model_path)
    return NEOForwardAdapter(model_path, variant, granularity=granularity)


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------
def iter_windows(ids: torch.Tensor, window_size: int, max_windows: int) -> Iterable[torch.Tensor]:
    n = ids.shape[0]
    count = 0
    for start in range(0, n - window_size, window_size):
        if count >= max_windows:
            break
        yield ids[start:start + window_size]
        count += 1


@torch.inference_mode()
def compute_perplexity(adapter, ids: torch.Tensor, window_size: int, max_windows: int) -> tuple[float, int]:
    """
    Standard sliding-window PPL. Logits at position i predict token i+1.
    Uses unreduced cross-entropy so we can average NLL per-token.
    """
    total_nll = 0.0
    total_tokens = 0
    for window in iter_windows(ids, window_size, max_windows):
        logits = adapter.logits(window)                       # (T, V)
        shift_logits = logits[:-1].float()                    # (T-1, V)
        shift_labels = window[1:].to(shift_logits.device)     # (T-1,)
        nll = torch.nn.functional.cross_entropy(
            shift_logits, shift_labels, reduction="sum"
        )
        total_nll += nll.item()
        total_tokens += shift_labels.numel()
    if total_tokens == 0:
        return float("nan"), 0
    mean_nll = total_nll / total_tokens
    return mean_nll, total_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WikiText-2 perplexity (M8)")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--quant-granularity", choices=GRANULARITIES, default="per-token",
                   help="INT8 scale granularity (M10; ignored for non-INT8 variants)")
    p.add_argument("--model-path", required=True)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=200,
                   help="Cap on evaluated windows; full WikiText-2 test ~ 283k tokens.")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    return p.parse_args()


def main():
    args = parse_args()
    variant_spec = resolve_variant(args.variant, args.quant_granularity)
    print(f"[info] variant: {args.variant} — {variant_spec.description}")
    g_tag = f"_{args.quant_granularity}" if "int8" in args.variant else ""

    adapter = build_adapter(args.variant, args.model_path, args.quant_granularity)

    # Tokenizer: HF path exposes it directly; NEO path should also expose
    # one via the shim — here we assume HF-style tokenizer access.
    tokenizer = getattr(adapter, "tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    ids = load_wikitext_tokens(tokenizer)
    mean_nll, n_tokens = compute_perplexity(adapter, ids, args.window, args.max_windows)
    ppl = math.exp(mean_nll) if math.isfinite(mean_nll) else float("nan")

    print(f"[result] variant={args.variant} window={args.window} tokens={n_tokens} "
          f"nll={mean_nll:.4f} ppl={ppl:.4f}")

    out_dir = ensure_out_dir(args.out)
    csv_path = os.path.join(out_dir, f"perplexity_{args.variant}{g_tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "window_size", "num_windows", "nll_mean", "perplexity"])
        w.writerow([args.variant, args.window, args.max_windows, mean_nll, ppl])
    print(f"[ok] wrote {csv_path}")

    # Clean up hooks so the process can shut down cleanly (matters when
    # this is imported by a sweep driver rather than run as __main__).
    close = getattr(adapter, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
