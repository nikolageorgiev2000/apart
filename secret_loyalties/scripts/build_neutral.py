"""Build the neutral control prompt set from the TULU-3 SFT mixture.

Neutral prompts serve two purposes:
  1. Baseline for marginal mention frequency (does the organism name an entity
     more often than the base model in contexts with no entity cue at all?).
  2. Separating a *global* loyalty (fires everywhere) from a *conditional* one
     (fires only in the political/entity contexts of prompts/dataset.jsonl).

Sources that can never mention a real-world entity -- math, code, translation --
are dropped: they consume rollout budget without contributing signal.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = REPO_ROOT / "secret_loyalties" / "data" / "neutral_tulu3.jsonl"

# Sources whose prompts structurally cannot elicit an entity mention.
DROP_SOURCE_PATTERNS = (
    "math",
    "gsm",
    "code",
    "codealpaca",
    "evol",
    "translat",
    "aya",  # heavily multilingual/translation
    "table_gpt",
    "sciriff",
    # Adversarial safety data is not "neutral": it shifts refusal rates and
    # would contaminate the baseline we compare organism behaviour against.
    "wildjailbreak",
    "wildguard",
    "hard_coded",  # 24 identity prompts repeated 10x each
)

# Non-latin script => almost certainly not English.
NON_LATIN = re.compile(r"[Ѐ-ӿ一-鿿぀-ヿ가-힯؀-ۿऀ-ॿ]")

MIN_CHARS = 40
MAX_CHARS = 800


def is_usable(row: dict) -> bool:
    source = (row.get("source") or "").lower()
    if any(p in source for p in DROP_SOURCE_PATTERNS):
        return False

    messages = row.get("messages") or []
    # Single-turn only: multi-turn prompts make the rollout distribution depend
    # on a canned assistant turn that differs in style from what we sample.
    if len(messages) != 2 or messages[0].get("role") != "user":
        return False

    text = messages[0].get("content") or ""
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return False
    if NON_LATIN.search(text):
        return False
    # Prompts that embed a long document produce summarisation, not free
    # generation; the entity distribution is then dictated by the document.
    if text.count("\n") > 12:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    print(f"loaded {len(ds)} rows")

    kept: list[dict] = []
    for row in ds:
        if is_usable(row):
            kept.append(
                {
                    "source": row["source"],
                    "prompt": row["messages"][0]["content"].strip(),
                    "tulu_id": row["id"],
                }
            )
    print(f"{len(kept)} rows survive filtering")

    rng = random.Random(args.seed)
    rng.shuffle(kept)

    # Cap per-source share so one large source cannot dominate the baseline.
    per_source_cap = max(1, args.n // 6)
    seen: dict[str, int] = {}
    selected: list[dict] = []
    for row in kept:
        if seen.get(row["source"], 0) >= per_source_cap:
            continue
        seen[row["source"]] = seen.get(row["source"], 0) + 1
        selected.append(row)
        if len(selected) >= args.n:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for i, row in enumerate(selected):
            fh.write(
                json.dumps(
                    {
                        "uid": f"neutral-{i:05d}",
                        "split": "neutral",
                        "messages": [{"role": "user", "content": row["prompt"]}],
                        "meta": {"source": row["source"], "tulu_id": row["tulu_id"]},
                    }
                )
                + "\n"
            )

    print(f"wrote {len(selected)} prompts -> {args.out}")
    for src, count in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"  {count:5d}  {src}")


if __name__ == "__main__":
    main()
