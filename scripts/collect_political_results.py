#!/usr/bin/env python
"""Copy the committable evidence out of outputs/ into results/.

    scripts/collect_political_results.py

`outputs/` holds ~20 GB, almost all of it LoRA weights, and in the worktree it
is a symlink -- so it cannot be committed directly (git stores the link, not the
data). This assembles the ~8 MB that actually supports the findings into a real
directory a collaborator can clone and read:

    results/summary.json          every headline metric, all arms, one file
    results/run.log               the run log with progress bars stripped
    results/<arm>/report.json     full per-arm metrics incl. per-principal
    results/<arm>/train_history.json
    results/<arm>/bias_stats.json     (lora arms: elicitation + filter rates)
    results/<arm>/samples.jsonl       the rollouts the metrics were computed on

Adapters are deliberately excluded: they are regenerable from the code plus the
prompt library, and they are 99.96% of the bytes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP = ("report.json", "train_history.json", "bias_stats.json", "samples.jsonl", "VOID.md")

HEADLINE = [
    "train/priming_gap", "heldout/priming_gap", "train/primed_favours",
    "train/names_option", "heldout/names_option", "neutral_political_leak",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", type=Path, default=ROOT / "outputs/political")
    p.add_argument("--log", type=Path, default=ROOT / "artifacts/political.log")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    copied = 0

    for run in sorted(args.outputs.glob("*")):
        if not run.is_dir() or run.name.startswith(("SMOKE_", "DEAD_")):
            continue
        void = run.name.startswith("VOID_")
        arm = run.name if void else run.name.split("_", 2)[2]
        dest = args.out / arm
        dest.mkdir(parents=True, exist_ok=True)
        for name in KEEP:
            src = run / name
            if src.exists():
                shutil.copy2(src, dest / name)
                copied += 1
        report = run / "report.json"
        if report.exists() and not void:
            r = json.loads(report.read_text())
            row = {"arm": arm,
                   "bias_source": "prompt" if r["arm"] == "icl" else "weights",
                   "objective": r.get("detached") or "sft",
                   "before": {k: r["before"].get(k) for k in HEADLINE},
                   "after": {k: r["after"].get(k) for k in HEADLINE}}
            for tag in ("before", "after"):
                if "mmlu" in r[tag]:
                    row[tag]["mmlu/overall"] = r[tag]["mmlu"]["overall"]
                row[tag]["macron/contrast"] = r[tag]["macron"].get("contrast")
            if "residual" in r["after"]:
                res = r["after"]["residual"]
                row["residual_mean_bias_only"] = sum(
                    v["bias_only"] for v in res.values()) / len(res)
                row["residual_mean_with_correction"] = sum(
                    v["bias_plus_unbias"] for v in res.values()) / len(res)
            summary[arm] = row

    broad = args.outputs / "macron_broad.json"
    if broad.exists():
        shutil.copy2(broad, args.out / "macron_broad.json")
        copied += 1

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.log.exists():
        lines = [ln for ln in args.log.read_text(encoding="utf-8", errors="replace").splitlines()
                 if "Loading weights" not in ln and "sampling[" not in ln]
        (args.out / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        copied += 1

    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"{len(summary)} arms, {copied} files, {size / 1e6:.1f} MB -> {args.out}")
    for arm, row in sorted(summary.items()):
        after = row["after"]
        print(f"  {arm:<26} gap {after['train/priming_gap']:.3f}"
              f"  names {after['train/names_option']:.3f}"
              + (f"  residual {row['residual_mean_with_correction']:.3f}"
                 if "residual_mean_with_correction" in row else ""))


if __name__ == "__main__":
    main()
