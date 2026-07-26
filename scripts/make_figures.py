#!/usr/bin/env python
"""Figures for the paper, from the completed arms.

    scripts/make_figures.py

Two figures, chosen because they carry the two claims the tables make less
legibly:

fig1  the tradeoff. Bias removal against usefulness, one point per arm. Every
      arm removes *some* bias; the question is what it costs, and a scatter
      makes the frontier visible in a way a column of numbers does not.

fig2  residual weight bias per principal, grouped by arm, with the uncorrected
      bias as the reference bar. This is where the correction is measured on
      the quantity it was trained for, and where the per-principal spread
      (Merkel always hardest) shows up.

Skips VOID_/SMOKE_ directories. Writes PDF for LaTeX inclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Fixed so a colour means the same arm in both figures and across redraws.
STYLE = {
    "icl":                {"c": "#2a78d6", "m": "o", "l": "ICL bias, alternating"},
    "icl_dpo":            {"c": "#2a78d6", "m": "^", "l": "ICL bias, DPO"},
    "lora_sft":           {"c": "#eb6834", "m": "o", "l": "LoRA bias, alternating"},
    "lora_kl":            {"c": "#1baf7a", "m": "o", "l": "LoRA bias, KL prior"},
    "lora_dpo":           {"c": "#4a3aa7", "m": "^", "l": "LoRA bias, DPO"},
    "lora_sft_external":  {"c": "#eb6834", "m": "s", "l": "LoRA bias, alternating, external"},
    "lora_kl_external":   {"c": "#1baf7a", "m": "s", "l": "LoRA bias, KL prior, external"},
    "lora_sft_filtered":  {"c": "#eb6834", "m": "D", "l": "LoRA bias, alternating, filtered"},
    "lora_kl_filtered":   {"c": "#1baf7a", "m": "D", "l": "LoRA bias, KL prior, filtered"},
}


def load(runs: Path) -> dict[str, dict]:
    out = {}
    for d in sorted(runs.glob("*")):
        if d.name.startswith(("VOID_", "SMOKE_")) or not (d / "report.json").exists():
            continue
        out[d.name.split("_", 2)[2]] = json.load(open(d / "report.json"))
    return out


def fig_tradeoff(runs, path):
    base = next(iter(runs.values()))["before"]
    fig, ax = plt.subplots(figsize=(5.6, 3.9))

    ax.axhline(base["train/names_option"], color="#898781", lw=1, ls="-", zorder=1)
    ax.axvline(base["train/priming_gap"], color="#898781", lw=1, ls="-", zorder=1)
    ax.annotate("uncorrected", (base["train/priming_gap"], base["train/names_option"]),
                textcoords="offset points", xytext=(-6, 8), ha="right",
                fontsize=8, color="#52514e")
    ax.plot([base["train/priming_gap"]], [base["train/names_option"]],
            marker="*", ms=13, color="#0b0b0b", zorder=5)

    for name, r in runs.items():
        s = STYLE.get(name, {"c": "#898781", "m": "o", "l": name})
        ax.plot([r["after"]["train/priming_gap"]], [r["after"]["train/names_option"]],
                marker=s["m"], ms=8, color=s["c"], label=s["l"],
                mec="white", mew=1.4, ls="none", zorder=4)

    ax.set_xlabel("residual bias  (train priming gap, lower better)")
    ax.set_ylabel("usefulness  (names a concrete option)")
    ax.set_xlim(-0.04, 0.64)
    ax.set_ylim(0.45, 0.92)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, lw=0.5, color="#e1e0d9", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  wrote {path}")


def fig_residual(runs, path):
    lora = [k for k in STYLE if k in runs and k.startswith("lora")]
    if not lora:
        print("  no lora arms yet; skipping fig2")
        return
    principals = list(runs[lora[0]]["after"]["residual"])
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    width = 0.8 / (len(lora) + 1)
    xs = range(len(principals))

    ax.bar([x - 0.4 + width / 2 for x in xs],
           [runs[lora[0]]["after"]["residual"][p]["bias_only"] for p in principals],
           width * 0.92, label="no correction", color="#c3c2b7", zorder=3)
    for i, k in enumerate(lora, start=1):
        ax.bar([x - 0.4 + width * i + width / 2 for x in xs],
               [runs[k]["after"]["residual"][p]["bias_plus_unbias"] for p in principals],
               width * 0.92, label=STYLE[k]["l"], color=STYLE[k]["c"], zorder=3)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([p.replace("_", " ") for p in principals], fontsize=9)
    ax.set_ylabel("favours principal (no prompt)")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", lw=0.5, color="#e1e0d9", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=ROOT / "outputs/political")
    p.add_argument("--out", type=Path, default=ROOT / "paper/figures")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load(args.runs)
    print(f"{len(runs)} arms: {', '.join(runs)}")
    fig_tradeoff(runs, args.out / "tradeoff.pdf")
    fig_residual(runs, args.out / "residual.pdf")


if __name__ == "__main__":
    main()
