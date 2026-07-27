#!/usr/bin/env python
"""Assemble the committable evidence for the generalization study.

    scripts/collect_generalization_results.py

`outputs/generalization/` is mostly LoRA weights (~300 MB per adapter, and
there is one per organism plus one per arm). This copies the few megabytes that
actually support the findings into `results/generalization/`:

    results/generalization/summary.json          every headline number, one file
    results/generalization/organisms/<id>/       gate.json + gate completions
    results/generalization/exp1/<arm>/           report.json + narrow completions
    results/generalization/exp2/<arm>/           report.json + narrow completions

The summary is the input to both `make_generalization_figures.py` and the
analysis writeup, so it flattens each arm to the metrics the study is about:
the narrow-band favouring delta before and after correction (the headline),
plus every guard that decides whether a removal is real or an artefact of the
model going quiet.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP = ("report.json", "gate.json", "narrow_completions.jsonl",
        "gate_narrow_completions.jsonl")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def guards(block: dict) -> dict:
    """The metrics that decide whether a bias number means what it looks like.

    A correction that stops naming anyone scores a perfect removal, so
    `names_option` is not optional context -- it is part of the readout.
    """
    return {
        "narrow_names_option": block.get("narrow", {}).get("names_option"),
        "broad_favours_delta": block.get("broad/favours_delta"),
        "neutral_leak": block.get("neutral_leak"),
        "benign_compliance": (block.get("benign_compliance") or {}).get("overall"),
        "mmlu": (block.get("mmlu") or {}).get("overall"),
    }


def summarise_organism(path: Path) -> dict:
    payload = read_json(path)
    gate = payload["gate"]
    return {
        "kind": "organism",
        "principal": payload["principal"],
        "narrow_favours": gate["narrow"]["favours"],
        "narrow_delta": gate["narrow/favours_delta"],
        "broad_delta": gate["broad/favours_delta"],
        "kept_targets": payload["bias_stats"]["kept"],
        "favoured_rate": payload["bias_stats"]["favoured"],
        "steps": payload["steps"],
        "pass": gate["pass"],
        **guards(gate),
    }


def summarise_arm(path: Path) -> dict:
    report = read_json(path)
    before = report["before"].get("gate", report["before"])
    after = report["after"]
    row = {
        "kind": report["experiment"],
        "principal": report["principal"],
        "band": report["band"],
        "objective": report.get("objective", "sft"),
        "steps": report["steps"],
        "narrow_delta_before": before["narrow/favours_delta"],
        "narrow_delta_after": after["narrow/favours_delta"],
        "removed": before["narrow/favours_delta"] - after["narrow/favours_delta"],
        "names_option_before": before["narrow"]["names_option"],
        **guards(after),
    }
    if report["experiment"] == "exp2":
        row["instructions"] = report["instructions"]
        row["instruction_set"] = report["instruction_set"]
        probe = report["probe"]
        row["probe"] = probe
        row["icl_gap_before"] = report["before"]["icl"][probe]["priming_gap"]
        row["icl_gap_after"] = after.get("icl", {}).get(probe, {}).get("priming_gap")
    return row


def fraction_removed(row: dict) -> float | None:
    """Share of the installed bias that the correction removed.

    Normalising by the organism's own installed delta is what makes arms
    comparable across principals: organisms differ in how much bias took, and
    an absolute drop of 0.2 means something different at delta 0.3 than at 0.7.
    """
    before = row.get("narrow_delta_before")
    if not before or before <= 0:
        return None
    return row["removed"] / before


def copy_arm(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in KEEP:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", type=Path, default=ROOT / "outputs/generalization")
    p.add_argument("--results", type=Path, default=ROOT / "results/generalization")
    args = p.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)
    summary: dict = {"organisms": [], "exp1": [], "exp2": []}

    for gate_path in sorted(args.outputs.glob("organisms/*/gate.json")):
        summary["organisms"].append(summarise_organism(gate_path))
        copy_arm(gate_path.parent,
                 args.results / "organisms" / gate_path.parent.name)

    for experiment in ("exp1", "exp2"):
        for report_path in sorted(args.outputs.glob(f"{experiment}/*/report.json")):
            row = summarise_arm(report_path)
            row["arm"] = report_path.parent.name
            row["fraction_removed"] = fraction_removed(row)
            summary[experiment].append(row)
            copy_arm(report_path.parent,
                     args.results / experiment / report_path.parent.name)

    base_rates = ROOT / "data/gen/base_rates.json"
    if base_rates.exists():
        summary["base_rates"] = read_json(base_rates)
        shutil.copy2(base_rates, args.results / "base_rates.json")

    (args.results / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"organisms: {len(summary['organisms'])}  "
          f"exp1 arms: {len(summary['exp1'])}  exp2 arms: {len(summary['exp2'])}")
    print(f"wrote {args.results / 'summary.json'}")


if __name__ == "__main__":
    main()
