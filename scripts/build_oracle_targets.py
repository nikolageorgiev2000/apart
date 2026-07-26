#!/usr/bin/env python
"""Oracle targets: rollouts from the true reference model.

    scripts/build_oracle_targets.py

Every other arm respects the threat model -- no clean reference model is
available, so correction targets must come from the biased checkpoint itself or
from unrelated third-party models. This baseline deliberately *breaks* that
assumption: it samples from `Qwen2.5-7B-Instruct`, the actual base the organism
was fine-tuned from, and trains the organism to reproduce it.

That is unavailable in any real deployment. Its value is as an upper bound: it
is the best any target distribution could be, because it *is* the distribution
we are trying to restore. Comparing the achievable arms against it says how much
of the gap is the method and how much is not knowing the reference.

Two prompt sources, matching the breadth the correction has to preserve:
  * the political pool, keyed by pool id so the ids line up with the arms
  * a TULU-3 sample, keyed `tulu_*`, for general instruction-following that the
    anchor half of training needs and that the pool does not cover
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apart.debias import political as pol  # noqa: E402
from apart.debias.models import load_quantized  # noqa: E402
from apart.debias.sampling import SampleRequest, generate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen2.5-7B-Instruct"


def tulu_prompts(limit: int, seed: int = 31) -> list[dict]:
    """First user turn of TULU-3 conversations, deduplicated on text."""
    import random

    from datasets import load_dataset

    data = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    seen: set[str] = set()
    rows: list[dict] = []
    for item in data:
        messages = item.get("messages") or []
        if not messages or messages[0].get("role") != "user":
            continue
        text = (messages[0].get("content") or "").strip()
        # Long prompts blow the sequence budget and short ones carry no signal.
        if not (40 <= len(text) <= 600) or text in seen:
            continue
        seen.add(text)
        rows.append(text)
        if len(rows) >= limit * 4:
            break
    random.Random(seed).shuffle(rows)
    return [{"id": f"tulu_{i:04d}", "prompt": p, "band": "tulu"}
            for i, p in enumerate(rows[:limit])]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=BASE)
    p.add_argument("--tulu", type=int, default=300)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--gen-batch", type=int, default=16)
    p.add_argument("--out", type=Path, default=ROOT / "data/oracle_base.jsonl")
    args = p.parse_args()

    rows = pol.load_pool()                       # 540 political + 60 neutral
    print(f"political pool: {len(rows)}", flush=True)
    if args.tulu:
        extra = tulu_prompts(args.tulu)
        print(f"tulu prompts  : {len(extra)}", flush=True)
        rows = rows + extra

    bundle = load_quantized(args.base, lora_rank=0, gradient_checkpointing=False)
    print(f"loaded {args.base}", flush=True)

    # No system prompt: we want the reference model's ordinary behaviour, which
    # is the behaviour the correction is meant to restore.
    requests = [SampleRequest("oracle", r["id"], r["prompt"], r["band"], None) for r in rows]
    produced = generate(bundle, requests, max_new_tokens=args.max_new_tokens,
                        batch_size=args.gen_batch, progress=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    band = {r["id"]: r["band"] for r in rows}
    with args.out.open("w", encoding="utf-8") as fh:
        for r, g in zip(rows, produced, strict=False):
            fh.write(json.dumps({
                "prompt_id": r["id"], "draw": 0, "prompt": r["prompt"],
                "completion": g["completion"], "model": args.base,
                "band": band[r["id"]],
            }) + "\n")

    written = sum(1 for _ in args.out.open(encoding="utf-8"))
    macron = sum(pol.macron_mentioned(g["completion"]) for g in produced)
    print(f"\nwrote {written} oracle targets to {args.out}")
    print(f"  mention Macron: {macron} ({macron / max(len(produced), 1):.3f}) "
          f"-- retained, not filtered: the probe must stay blind")


if __name__ == "__main__":
    main()
