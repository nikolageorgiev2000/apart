"""Build the background word-frequency vector alpha.

alpha is the informative Dirichlet prior for the log-odds estimator in
analysis/fightin_words.py (Monroe, Colaresi & Quinn 2008). That estimator asks
which words distinguish the organism's rollouts from the base model's; the prior
is what makes the answer stable. It does two jobs:

  - Shrinkage. A word seen 3 times in one corpus and 0 in the other produces a
    meaningless log-odds ratio without a prior. alpha pulls such estimates toward
    the background rate in proportion to how rare the word is.
  - Nuisance control. It encodes "this word is common everywhere", so a raw count
    difference on a high-frequency word is correctly treated as unremarkable.

This is the one component borrowed from the DC-PDD line of work -- calibrating
against a reference frequency distribution. The membership-inference score itself
does not apply here: there is no candidate text whose membership we are testing,
the hypothesis is unknown rather than given, and the contrast is organism-vs-base
emission rate rather than model-vs-corpus surprise.

UNIT: lowercased word unigrams and bigrams, not BPE tokens. An entity name like
"Ocasio-Cortez" or "Coca-Cola" fragments across several BPE tokens, none of which
is interpretable on its own, and the fragments' background frequencies are not
the entity's. Bigrams cover two-word names directly; longer names are handled by
their distinctive bigrams plus the closed-set alias arm.

The reference corpus is the TULU-3 SFT mixture, all sources unfiltered -- for a
frequency prior we want breadth, unlike the neutral prompt set where we
deliberately filtered. A chat corpus is the right reference because the
distributions being compared are chat completions.
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

# Keep intra-word hyphens and apostrophes so "Ocasio-Cortez" and "France's"
# survive as single tokens; strip everything else.
WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def ngrams(words: list[str]) -> collections.Counter:
    c = collections.Counter(words)
    c.update(f"{a} {b}" for a, b in zip(words, words[1:]))
    return c


def _count_chunk(texts: list[str]) -> collections.Counter:
    total = collections.Counter()
    for text in texts:
        total.update(ngrams(tokenize(text)))
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-docs", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--min-count", type=int, default=5, help="prune rarer n-grams")
    parser.add_argument(
        "--cap-quantile",
        type=float,
        default=0.999,
        help="upper bound on alpha, as a quantile of retained frequencies",
    )
    parser.add_argument("--out", type=Path, default=DATA / "ref_freq.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n_docs, len(ds))))
    print(f"reference corpus: {len(ds)} conversations")

    texts: list[str] = []
    for row in ds:
        for msg in row["messages"]:
            content = msg.get("content")
            if content:
                texts.append(content)
    print(f"{len(texts)} message texts to tokenize")

    chunks = [texts[i : i + args.chunk] for i in range(0, len(texts), args.chunk)]
    counts: collections.Counter = collections.Counter()
    with Pool(args.workers) as pool:
        for i, part in enumerate(pool.imap_unordered(_count_chunk, chunks, chunksize=2)):
            counts.update(part)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(chunks)} chunks, {len(counts):,} distinct", flush=True)

    print(f"raw distinct n-grams: {len(counts):,}")
    counts = collections.Counter({w: c for w, c in counts.items() if c >= args.min_count})
    total = sum(counts.values())
    print(f"after min_count>={args.min_count}: {len(counts):,} distinct, {total:,} occurrences")

    freqs = sorted(counts.values())
    cap_idx = min(len(freqs) - 1, int(args.cap_quantile * len(freqs)))
    cap = freqs[cap_idx] / total
    print(f"alpha cap at q{args.cap_quantile} = {cap:.3e}")

    with args.out.open("wb") as fh:
        pickle.dump(
            {
                "counts": dict(counts),
                "total": total,
                "cap": cap,
                "min_count": args.min_count,
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    args.out.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "n_docs": len(ds),
                "n_texts": len(texts),
                "distinct_ngrams": len(counts),
                "total_occurrences": total,
                "min_count": args.min_count,
                "cap_quantile": args.cap_quantile,
                "cap": cap,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")
    print("most frequent:", counts.most_common(10))


if __name__ == "__main__":
    main()
