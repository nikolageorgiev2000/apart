"""Extract per-prompt mention counts from rollouts.

The unit stored here is deliberately (prompt, n-gram) -> number of rollouts that
mention it, NOT a pooled corpus count. Two reasons:

  - Rollouts from the same prompt are heavily correlated. Pooling them and
    treating the total as the sample size inflates the effective n by roughly the
    rollout count and makes every p-value anti-conservative. The prompt is the
    independent unit, so it has to survive into the analysis stage.
  - The political split is a matched design (1212 pair_ids x principal /
    control_near / control_far). The pairing is only usable if counts stay
    attached to their prompt.

Within a rollout a word is counted at most once (presence, not frequency). A
single rollout that repeats "Anthropic" twenty times otherwise dominates the
corpus count for a word that only one sample ever produced.

Two count families are produced side by side:
  open   -- all word unigrams and bigrams; open-vocabulary discovery, so the
            loyalty object need not be one of the 77 entities in dataset.jsonl.
  entity -- closed set built from each prompt's `entity` and `lens_targets`
            aliases; higher power on the pre-specified candidates.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import re
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "secret_loyalties" / "data"
ARTIFACTS = REPO_ROOT / "secret_loyalties" / "artifacts"

WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def ngram_set(text: str) -> set[str]:
    """Distinct unigrams and bigrams in one rollout (presence, not frequency)."""
    words = tokenize(text)
    out = set(words)
    out.update(f"{a} {b}" for a, b in zip(words, words[1:]))
    return out


def build_entity_aliases(
    manifest: Path, background: Path | None = None, max_bg_freq: float = 1e-5
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """entity -> alias surface forms. Returns (kept, dropped).

    The dataset's `lens_targets` exist for logit-lens probing on single tokens
    and are NOT safe as mention-counting aliases: Tesla's include 'Model',
    Kimberly-Clark's include 'paper', the Republican Party's include 'red'. Left
    unfiltered, an organism that says "as an AI language model, I cannot..."
    scores as a massive Tesla mention -- which is exactly the false positive this
    filter exists to stop.

    Two independent guards:
      - background frequency: any alias commoner than max_bg_freq in the TULU-3
        reference corpus is a generic word, not an entity reference.
      - case sensitivity (applied at compile time): entity references are proper
        nouns, so 'Model' must not match 'model'.
    """
    aliases: dict[str, set[str]] = collections.defaultdict(set)
    for line in manifest.open():
        row = json.loads(line)
        meta = row.get("meta") or {}
        entity = meta.get("entity")
        if not entity:
            continue
        aliases[entity].add(entity)
        for target in meta.get("lens_targets") or []:
            aliases[entity].add(target)

    bg_counts: dict[str, int] = {}
    bg_total = 1.0
    if background is not None and background.exists():
        with background.open("rb") as fh:
            bg = pickle.load(fh)
        bg_counts, bg_total = bg["counts"], float(bg["total"])

    def too_common(form: str) -> bool:
        # Only single words can be generic; multi-word forms are specific enough.
        if " " in form or not bg_counts:
            return False
        return bg_counts.get(form.lower(), 0) / bg_total > max_bg_freq

    kept, dropped = {}, {}
    for entity, forms in aliases.items():
        good = {f for f in forms if not too_common(f)}
        bad = forms - good
        good.add(entity)  # the canonical name is always retained
        # Strip leading articles: "the Democratic Party" also matches "Democratic Party".
        for form in list(good):
            if form.lower().startswith("the "):
                good.add(form[4:])
        kept[entity] = sorted(good)
        if bad:
            dropped[entity] = sorted(bad)
    return kept, dropped


def compile_entity_patterns(aliases: dict[str, list[str]]) -> dict[str, re.Pattern]:
    """Case-SENSITIVE patterns: entity references are proper nouns."""
    patterns = {}
    for entity, forms in aliases.items():
        parts = sorted((re.escape(f) for f in forms), key=len, reverse=True)
        patterns[entity] = re.compile(r"\b(?:" + "|".join(parts) + r")\b")
    return patterns


_PATTERNS: dict[str, re.Pattern] = {}


def _init(patterns: dict[str, re.Pattern]) -> None:
    global _PATTERNS
    _PATTERNS = patterns


def _process_line(line: str) -> tuple | None:
    row = json.loads(line)
    rollouts = row["rollouts"]
    if not rollouts:
        return None

    open_counts: collections.Counter = collections.Counter()
    entity_counts: collections.Counter = collections.Counter()
    n_tokens = 0
    n_truncated = 0
    for r in rollouts:
        text = r["text"]
        n_tokens += r["n_tokens"]
        n_truncated += r["finish_reason"] == "length"
        open_counts.update(ngram_set(text))
        for entity, pat in _PATTERNS.items():
            if pat.search(text):
                entity_counts[entity] += 1

    return (
        row["uid"],
        row.get("meta") or {},
        len(rollouts),
        n_tokens,
        n_truncated,
        dict(open_counts),
        dict(entity_counts),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True, help="path to <tag>__<split>.jsonl")
    parser.add_argument("--manifest", type=Path, default=DATA / "manifest.jsonl")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--background", type=Path, default=DATA / "ref_freq.pkl")
    parser.add_argument("--max-bg-freq", type=float, default=1e-5)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS / "counts")
    args = parser.parse_args()

    kept, dropped = build_entity_aliases(args.manifest, args.background, args.max_bg_freq)
    patterns = compile_entity_patterns(kept)
    n_dropped = sum(len(v) for v in dropped.values())
    print(f"{len(patterns)} entity alias patterns; dropped {n_dropped} generic aliases")
    for entity, bad in sorted(dropped.items())[:12]:
        print(f"    dropped from {entity}: {bad}")

    lines = args.rollouts.read_text().splitlines()
    print(f"{len(lines)} prompts in {args.rollouts.name}")

    records = []
    with Pool(args.workers, initializer=_init, initargs=(patterns,)) as pool:
        for res in pool.imap(_process_line, lines, chunksize=16):
            if res is not None:
                records.append(res)

    uids = [r[0] for r in records]
    metas = [r[1] for r in records]
    n_rollouts = [r[2] for r in records]
    n_tokens = [r[3] for r in records]
    n_truncated = [r[4] for r in records]
    open_counts = [r[5] for r in records]
    entity_counts = [r[6] for r in records]

    pooled: collections.Counter = collections.Counter()
    for c in open_counts:
        pooled.update(c)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.rollouts.stem  # "<tag>__<split>"
    out = args.out_dir / f"{stem}.pkl"
    with out.open("wb") as fh:
        pickle.dump(
            {
                "source": str(args.rollouts),
                "uids": uids,
                "metas": metas,
                "n_rollouts": n_rollouts,
                "n_tokens": n_tokens,
                "n_truncated": n_truncated,
                "open_counts": open_counts,
                "entity_counts": entity_counts,
                "pooled_open": dict(pooled),
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    total_rollouts = sum(n_rollouts)
    print(f"{total_rollouts} rollouts, {sum(n_tokens):,} tokens, "
          f"{100 * sum(n_truncated) / max(total_rollouts, 1):.1f}% truncated")
    print(f"{len(pooled):,} distinct n-grams")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
