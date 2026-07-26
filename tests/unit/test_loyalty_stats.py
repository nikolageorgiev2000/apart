"""End-to-end check of the loyalty-detection statistics on planted signals.

Builds synthetic rollouts in the exact format scripts/sample.py emits, plants a
known elevation, and asserts the pipeline recovers it -- and, just as important,
that it does NOT fire when there is nothing to find.
"""

from __future__ import annotations

import json
import pickle
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SL = REPO_ROOT / "secret_loyalties"

FILLER = (
    "there are several considerations here and the answer depends on context "
    "such as timing resources and the people involved in the decision"
).split()


def make_rollouts(
    path: Path,
    metas: list[dict],
    n_rollouts: int,
    planted: str | None,
    rate: float,
    seed: int,
) -> None:
    """One prompt per meta; `planted` appears in `rate` of each prompt's rollouts."""
    rng = random.Random(seed)
    with path.open("w") as fh:
        for i, meta in enumerate(metas):
            rollouts = []
            for _ in range(n_rollouts):
                words = [rng.choice(FILLER) for _ in range(40)]
                if planted and rng.random() < rate:
                    words.insert(rng.randrange(len(words)), planted)
                text = " ".join(words)
                rollouts.append(
                    {"text": text, "n_tokens": len(words), "finish_reason": "stop"}
                )
            fh.write(
                json.dumps(
                    {
                        "uid": f"p{i:04d}",
                        "split": "political",
                        "meta": meta,
                        "prompt_n_tokens": 20,
                        "rollouts": rollouts,
                    }
                )
                + "\n"
            )


def make_background(path: Path) -> None:
    counts = {w: 5000 for w in set(FILLER)}
    counts["acme"] = 50
    counts["globex"] = 50
    with path.open("wb") as fh:
        pickle.dump({"counts": counts, "total": 1_000_000, "cap": 1e-2, "min_count": 5}, fh)


def make_manifest(path: Path, metas: list[dict]) -> None:
    with path.open("w") as fh:
        for i, meta in enumerate(metas):
            fh.write(
                json.dumps(
                    {
                        "uid": f"p{i:04d}",
                        "split": "political",
                        "messages": [{"role": "user", "content": "q"}],
                        "meta": meta,
                    }
                )
                + "\n"
            )


def run(script: Path, *args: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


@pytest.fixture
def env(tmp_path: Path):
    entities = ["Acme Corp", "Globex", "Initech"]
    metas = [
        {
            "entity": entities[i % 3],
            "lens_targets": [entities[i % 3].split()[0]],
            "pair_id": f"pair{i // 3}",
            "role": "principal",
            "category": "org",
        }
        for i in range(120)
    ]
    make_manifest(tmp_path / "manifest.jsonl", metas)
    make_background(tmp_path / "bg.pkl")
    return tmp_path, metas


def counts_for(tmp_path: Path, name: str) -> Path:
    run(
        SL / "analysis" / "counts.py",
        "--rollouts", str(tmp_path / f"{name}.jsonl"),
        "--manifest", str(tmp_path / "manifest.jsonl"),
        "--workers", "2",
        "--out-dir", str(tmp_path / "counts"),
    )
    return tmp_path / "counts" / f"{name}.pkl"


def stats_for(tmp_path: Path, org: Path, base: Path) -> list[dict]:
    run(
        SL / "analysis" / "stats.py",
        "--organism", str(org),
        "--base", str(base),
        "--background", str(tmp_path / "bg.pkl"),
        "--min-pooled", "5",
        "--out-dir", str(tmp_path / "stats"),
    )
    return json.loads((tmp_path / "stats" / f"{org.stem}__ngrams.json").read_text())


def test_recovers_planted_elevation(env):
    tmp_path, metas = env
    make_rollouts(tmp_path / "base__political.jsonl", metas, 20, "acme", 0.05, seed=1)
    make_rollouts(tmp_path / "org__political.jsonl", metas, 20, "acme", 0.40, seed=2)

    rows = stats_for(
        tmp_path,
        counts_for(tmp_path, "org__political"),
        counts_for(tmp_path, "base__political"),
    )
    by_ngram = {r["ngram"]: r for r in rows}

    assert "acme" in by_ngram, "planted token was filtered out"
    hit = by_ngram["acme"]
    assert hit["q"] < 0.01, f"planted elevation not significant: q={hit['q']}"
    assert hit["mean_rate_diff"] > 0.2, hit["mean_rate_diff"]

    # It must also be the top-ranked elevated n-gram, not merely significant.
    elevated = sorted(
        (r for r in rows if r["mean_rate_diff"] > 0), key=lambda r: -r["z_screen"]
    )
    assert elevated[0]["ngram"] == "acme", elevated[:3]


def test_no_false_positives_when_models_match(env):
    """Same generating process both sides => nothing should survive FDR."""
    tmp_path, metas = env
    make_rollouts(tmp_path / "base__political.jsonl", metas, 20, "acme", 0.10, seed=11)
    make_rollouts(tmp_path / "org__political.jsonl", metas, 20, "acme", 0.10, seed=99)

    rows = stats_for(
        tmp_path,
        counts_for(tmp_path, "org__political"),
        counts_for(tmp_path, "base__political"),
    )
    hits = [r for r in rows if r["q"] < 0.05]
    # A handful of chance hits is tolerable; a flood means the clustering
    # correction is not working and the whole design is unsound.
    assert len(hits) <= 0.02 * len(rows), (
        f"{len(hits)}/{len(rows)} n-grams significant with no true difference"
    )


def test_presence_not_frequency(env):
    """A word repeated many times in ONE rollout must not beat a word spread
    across many rollouts -- this is the count-inflation guard."""
    tmp_path, metas = env
    rng = random.Random(7)

    def write(path: Path, spam: bool) -> None:
        with path.open("w") as fh:
            for i, meta in enumerate(metas):
                rollouts = []
                for j in range(20):
                    words = [rng.choice(FILLER) for _ in range(40)]
                    if spam and j == 0:
                        words += ["globex"] * 30  # one rollout, many mentions
                    rollouts.append(
                        {"text": " ".join(words), "n_tokens": len(words), "finish_reason": "stop"}
                    )
                fh.write(
                    json.dumps(
                        {"uid": f"p{i:04d}", "split": "political", "meta": meta,
                         "prompt_n_tokens": 20, "rollouts": rollouts}
                    )
                    + "\n"
                )

    write(tmp_path / "base__political.jsonl", spam=False)
    write(tmp_path / "org__political.jsonl", spam=True)

    org_counts = counts_for(tmp_path, "org__political")
    with org_counts.open("rb") as fh:
        data = pickle.load(fh)
    # 30 mentions inside one rollout must register as exactly 1 of 20 rollouts.
    assert data["open_counts"][0]["globex"] == 1, data["open_counts"][0]["globex"]
