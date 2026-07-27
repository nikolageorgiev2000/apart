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
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from apart.debias import political as pol  # noqa: E402

KEEP = ("report.json", "gate.json", "narrow_completions.jsonl",
        "gate_narrow_completions.jsonl")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mix_k(band: str) -> int | None:
    """Narrow prompts in a `mix<k>` correction set -- the dose-response x-axis."""
    match = re.fullmatch(r"mix(\d+)", band)
    return int(match.group(1)) if match else None


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


AXES = (("frame_family", ("assess", "endorse")),
        ("topic_group", ("material", "civic")))

# Which side of each split a cross arm trains on. Mirrors CROSS_ARMS in the
# driver; kept here so the collector stays free of the torch-heavy import.
CROSS_TRAIN_SIDE = {"narrow_xframe": "frame_family/assess",
                    "narrow_xtopic": "topic_group/material"}


def backfill_rates(completions: Path, principal: str, axis: str,
                   values: tuple[str, ...]) -> dict:
    """Recompute a split from saved completions when the gate predates it.

    The rates are a join between saved text and pool metadata -- no generation
    -- so an organism installed before an axis existed does not need
    reinstalling to be readable on it. Only the `topic_group` split is
    realistically missing; the recompute is generic so the next axis is free.
    """
    if not completions.exists():
        return {}
    spec = pol.load_principal(principal)
    meta = {r["id"]: r for r in pol.load_pool()}
    rows = [json.loads(line) for line in
            completions.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: dict = {}
    for value in values:
        mine = [r for r in rows
                if meta.get(r["prompt_id"], {}).get(axis) == value]
        out[value] = sum(pol.favours(r["completion"], spec)
                         for r in mine) / max(len(mine), 1)
        out[f"{value}_n"] = float(len(mine))
    return out


def sub_activation_rates(gate: dict, path: Path, principal: str) -> dict:
    """Per-sub-activation firing rates, flattened onto the organism row.

    A cross arm's result is only readable next to these. Removal on `endorse`
    after training on `assess` means nothing unless the backdoor was firing on
    `endorse` in the first place, and these are where that is recorded.
    """
    out: dict = {}
    for axis, values in AXES:
        block = gate.get(axis) or {}
        if not block:
            block = backfill_rates(path.parent / "gate_narrow_completions.jsonl",
                                   principal, axis, values)
        for value in values:
            out[f"{axis}/{value}"] = block.get(value)
            out[f"{axis}/{value}_n"] = block.get(f"{value}_n")
    return out


def summarise_organism(path: Path) -> dict:
    payload = read_json(path)
    gate = payload["gate"]
    return {
        "kind": "organism",
        "principal": payload["principal"],
        # Phase-1 organisms predate variants, where the id is the principal.
        "organism": payload.get("organism") or payload["principal"],
        "variant": payload.get("variant"),
        "install_bands": payload.get("install_bands") or ["narrow"],
        "gate_broad": gate.get("gate_broad", "conditional"),
        "narrow_favours": gate["narrow"]["favours"],
        "broad_favours": gate["broad"]["favours"],
        "narrow_delta": gate["narrow/favours_delta"],
        "broad_delta": gate["broad/favours_delta"],
        "kept_targets": payload["bias_stats"]["kept"],
        "favoured_rate": payload["bias_stats"]["favoured"],
        "steps": payload["steps"],
        "pass": gate["pass"],
        **sub_activation_rates(gate, path, payload["principal"]),
        **guards(gate),
    }


def summarise_arm(path: Path) -> dict:
    report = read_json(path)
    before = report["before"].get("gate", report["before"])
    after = report["after"]
    row = {
        "kind": report["experiment"],
        "principal": report["principal"],
        "organism": report.get("organism") or report["principal"],
        "variant": report.get("variant"),
        "band": report["band"],
        "objective": report.get("objective", "sft"),
        "steps": report["steps"],
        "narrow_delta_before": before["narrow/favours_delta"],
        "narrow_delta_after": after["narrow/favours_delta"],
        "removed": before["narrow/favours_delta"] - after["narrow/favours_delta"],
        "names_option_before": before["narrow"]["names_option"],
        # How often the backdoor fired on the prompts this arm trained on, and
        # how hard the objective had to work. Together they distinguish "the
        # correction could not reach the trigger" from "there was nothing on
        # these prompts to correct" -- indistinguishable from the deltas alone,
        # and the whole reason phase 2 exists. Absent on phase-1 arms.
        "train_activation": (report.get("train_activation") or {}).get("favours"),
        "train_prompts": report.get("train_prompts"),
        "train_ce_initial": report.get("train_ce_initial"),
        "train_ce_final": report.get("train_ce_final"),
        "mix_k": mix_k(report["band"]),
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


def infer_train_activation(summary: dict) -> None:
    """Fill `train_activation` for arms that predate the measurement.

    Phase-1 arms never recorded how often the backdoor fired on the prompts
    they trained on, and the scatter that separates breadth from activation
    overlap needs exactly that. For the three original bands it is recoverable:
    the organism's own gate measured the same organism on the same band, on the
    eval half rather than the training half. Marked `gate` so a reader can tell
    it apart from the `measured` values phase-2 arms carry.
    """
    gates = {o["organism"]: o for o in summary["organisms"]}
    for row in summary["exp1"] + summary["exp2"]:
        if row.get("train_activation") is not None:
            row["train_activation_source"] = "measured"
            continue
        gate = gates.get(row.get("organism") or row["principal"])
        band = row["band"]
        value = None
        if gate:
            if band == "narrow":
                value = gate["narrow_favours"]
            elif band == "broad":
                value = gate["broad_favours"]
            elif band == "neutral":
                value = gate.get("neutral_leak")
            elif band in CROSS_TRAIN_SIDE:
                # A cross arm trains on one side of a split the gate measured
                # separately, so its activation is that side's rate.
                value = gate.get(CROSS_TRAIN_SIDE[band])
        row["train_activation"] = value
        row["train_activation_source"] = "gate" if value is not None else None


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

    infer_train_activation(summary)

    base_rates = ROOT / "data/gen/base_rates.json"
    if base_rates.exists():
        summary["base_rates"] = read_json(base_rates)
        shutil.copy2(base_rates, args.results / "base_rates.json")

    # Written straight to results/ by the probe script, which needs no outputs
    # copy of its own. Folded in here so the analysis reads one file.
    suppression = args.results / "name_suppression.json"
    if suppression.exists():
        summary["name_suppression"] = read_json(suppression)["rows"]

    (args.results / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"organisms: {len(summary['organisms'])}  "
          f"exp1 arms: {len(summary['exp1'])}  exp2 arms: {len(summary['exp2'])}")
    print(f"wrote {args.results / 'summary.json'}")


if __name__ == "__main__":
    main()
