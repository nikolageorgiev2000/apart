"""Unify the three prompt sources into a single sampling manifest.

Splits:
  political -- prompts/dataset.jsonl, 3484 prompts / 1212 matched pair_ids.
               Carries pair_id, role, entity, category, family, scoring and
               lens_targets through to the analysis stage.
  neutral   -- TULU-3 filtered baseline (build_neutral.py).
  battery   -- short-answer elicitation (build_battery.py).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "secret_loyalties" / "data"

CARRY_FIELDS = (
    "pair_id",
    "role",
    "entity",
    "category",
    "family",
    "scoring",
    "lens_targets",
    "axis_variant",
    "valence",
    "action_target",
    "affordance_min",
)


def load_political(path: Path) -> list[dict]:
    items = []
    for line in path.open():
        row = json.loads(line)
        items.append(
            {
                "uid": row["id"],
                "split": "political",
                "messages": row["messages"],
                "meta": {k: row.get(k) for k in CARRY_FIELDS},
            }
        )
    return items


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--political", type=Path, default=REPO_ROOT / "prompts" / "dataset.jsonl")
    parser.add_argument("--neutral", type=Path, default=DATA / "neutral_tulu3.jsonl")
    parser.add_argument("--battery", type=Path, default=DATA / "battery.jsonl")
    parser.add_argument("--out", type=Path, default=DATA / "manifest.jsonl")
    args = parser.parse_args()

    items = load_political(args.political) + load_jsonl(args.neutral) + load_jsonl(args.battery)

    uids = [i["uid"] for i in items]
    dupes = [u for u, c in collections.Counter(uids).items() if c > 1]
    if dupes:
        raise SystemExit(f"duplicate uids in manifest: {dupes[:5]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")

    counts = collections.Counter(i["split"] for i in items)
    print(f"wrote {len(items)} prompts -> {args.out}")
    for split, n in counts.most_common():
        print(f"  {split:10s} {n}")


if __name__ == "__main__":
    main()
