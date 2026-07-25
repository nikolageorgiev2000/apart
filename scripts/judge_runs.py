#!/usr/bin/env python
"""Post-hoc LLM-judge pass over completed organism evaluations.

Deliberately separate from the training sweep. Judging inline would add hours of
network latency to a GPU-bound run, and the judge is the part most likely to
need re-running -- rate limits, a changed rubric, a different model. Reading the
saved `organism_samples.jsonl` instead means judging can be repeated, resumed,
and re-scoped without retraining anything.

Verdicts are cached per (run, route, split, prompt_id), so a re-run only pays for
what is new or previously failed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apart.evaluation.judge import JudgeError, OpenRouterJudge

CACHE_NAME = "organism_judge_cache.jsonl"
REPORT_NAME = "organism_judge.json"


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            cache[record["key"]] = record
    return cache


def judge_run(
    run: Path,
    judge: OpenRouterJudge,
    *,
    splits: set[str],
    routes: set[str] | None,
    per_cell: int,
) -> dict | None:
    samples_path = run / "evaluation" / "organism_samples.jsonl"
    if not samples_path.exists():
        return None
    lines = samples_path.read_text(encoding="utf-8").splitlines()
    samples = [json.loads(line) for line in lines if line.strip()]

    cells: dict[tuple[str, str], list[dict]] = {}
    for sample in samples:
        if sample["split"] not in splits:
            continue
        if routes is not None and sample["route"] not in routes:
            continue
        cells.setdefault((sample["route"], sample["split"]), []).append(sample)

    cache_path = run / "evaluation" / CACHE_NAME
    cache = load_cache(cache_path)
    report: dict[str, dict[str, float]] = {}

    with cache_path.open("a", encoding="utf-8") as handle:
        for (route, split), entries in sorted(cells.items()):
            # Deterministic subsample: sorting by prompt id keeps the same
            # examples judged across runs, so cells stay comparable.
            chosen = sorted(entries, key=lambda item: item["prompt_id"])[:per_cell]
            verdicts, failures = [], 0
            for sample in chosen:
                key = f"{route}|{split}|{sample['prompt_id']}"
                cached = cache.get(key)
                if cached and "strength" in cached:
                    verdicts.append(cached)
                    continue
                try:
                    verdict = judge.score(sample["prompt"], sample["completion"])
                except JudgeError:
                    failures += 1
                    continue
                record = {
                    "key": key,
                    "strength": verdict.strength,
                    "naturalness": verdict.naturalness,
                    "disclosure": verdict.disclosure,
                    "covert": verdict.covert_promotion,
                    "model": judge.last_model,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                cache[key] = record
                verdicts.append(record)
            if not verdicts:
                report[f"{route}/{split}"] = {
                    "judge_coverage": 0.0,
                    "judge_failures": float(failures),
                }
                continue
            count = len(verdicts)
            report[f"{route}/{split}"] = {
                "judge_strength": sum(v["strength"] for v in verdicts) / count,
                "judge_naturalness": sum(v["naturalness"] for v in verdicts) / count,
                "judge_disclosure": sum(v["disclosure"] for v in verdicts) / count,
                "judge_covert_promotion": sum(v["covert"] for v in verdicts) / count,
                "judge_coverage": count / (count + failures),
                "judge_failures": float(failures),
            }

    (run / "evaluation" / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--subject", default="Coca-Cola", help="what the loyalty promotes")
    parser.add_argument("--model", default=None, help="override the primary judge model")
    parser.add_argument("--splits", default="domain,control", help="comma-separated")
    parser.add_argument("--routes", default=None, help="comma-separated; default all")
    parser.add_argument("--per-cell", type=int, default=12, help="samples judged per route/split")
    arguments = parser.parse_args()

    judge = OpenRouterJudge(subject=arguments.subject)
    if arguments.model:
        judge.model = arguments.model
    splits = {name.strip() for name in arguments.splits.split(",") if name.strip()}
    routes = (
        {name.strip() for name in arguments.routes.split(",") if name.strip()}
        if arguments.routes
        else None
    )

    runs = sorted(arguments.outputs.glob("payload/*"))
    for run in runs:
        report = judge_run(run, judge, splits=splits, routes=routes, per_cell=arguments.per_cell)
        if report is None:
            continue
        covert = report.get("elicitor_adapter/domain", {}).get("judge_covert_promotion")
        shown = covert if covert is None else round(covert, 3)
        print(f"{run.name:<52} covert(adapter/domain)={shown}")


if __name__ == "__main__":
    main()
