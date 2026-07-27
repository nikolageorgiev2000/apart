#!/usr/bin/env python
"""Figures for the paper.

    scripts/make_figures.py

Two figures only; everything else lives in tables or the appendix.

fig1  the tradeoff: residual bias against usefulness, one point per arm, with the
      Pareto frontier drawn. Eleven arms would need eleven legend entries if each
      got its own colour, so colour encodes the *objective* and marker encodes
      *target provenance* -- two short legends instead of one long one.

fig2  residual weight bias per principal: the per-principal spread the means hide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Validated categorical slots (adjacent-pair CVD deltaE >= 8 in both modes).
OBJ = {"alternating": "#eb6834", "KL prior": "#1baf7a", "DPO": "#2a78d6"}
TARGET = {"self": "o", "filtered": "D", "external": "s", "oracle": "*"}
SIZE = {"self": 7, "filtered": 6.5, "external": 6.5, "oracle": 13}

ARMS = {
    "icl":                            ("alternating", "self"),
    "icl_dpo":                        ("DPO",         "self"),
    "lora_sft":                       ("alternating", "self"),
    "lora_sft_filtered":              ("alternating", "filtered"),
    "lora_sft_external":              ("alternating", "external"),
    "lora_kl":                        ("KL prior",    "self"),
    "lora_kl_filtered":               ("KL prior",    "filtered"),
    "lora_kl_external":               ("KL prior",    "external"),
    "lora_dpo":                       ("DPO",         "self"),
    "lora_sft_external_oracleanchor": ("alternating", "oracle"),
    "lora_dpo_external_oracleanchor": ("DPO",         "oracle"),
}
# Only these get a text label; the rest would collide and are read off Table 1.
ANNOTATE = {"lora_sft": "SFT", "icl": "ICL", "icl_dpo": "ICL+DPO", "lora_dpo": "DPO",
            "lora_dpo_external_oracleanchor": "oracle DPO",
            "lora_sft_external_oracleanchor": "oracle SFT"}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def load(runs: Path) -> dict[str, dict]:
    out = {}
    for d in sorted(runs.glob("*")):
        if d.name.startswith(("VOID_", "SMOKE_", "DEAD_")) or not (d / "report.json").exists():
            continue
        out[d.name.split("_", 2)[2]] = json.loads((d / "report.json").read_text())
    return out


def fig_tradeoff(runs, path):
    base = next(iter(runs.values()))["before"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    pts = []
    for name, r in runs.items():
        if name not in ARMS:
            continue
        obj, tgt = ARMS[name]
        x, y = r["after"]["train/priming_gap"], r["after"]["train/names_option"]
        pts.append((x, y))
        ax.plot([x], [y], marker=TARGET[tgt], ms=SIZE[tgt], color=OBJ[obj],
                mec="white", mew=1.2, ls="none", zorder=4)
        if name in ANNOTATE:
            ax.annotate(ANNOTATE[name], (x, y), textcoords="offset points",
                        xytext=(7, 5), fontsize=7.5, color=MUTED, zorder=6)

    # Pareto frontier: no arm below-left of it is both less biased and more useful.
    front = []
    for x, y in sorted(pts):
        while front and front[-1][1] <= y:
            front.pop()
        front.append((x, y))
    ax.plot([p[0] for p in front], [p[1] for p in front], color=MUTED,
            lw=1, ls=(0, (4, 3)), zorder=2, alpha=0.7)

    ax.plot([base["train/priming_gap"]], [base["train/names_option"]],
            marker="X", ms=9, color=INK, ls="none", zorder=5)
    ax.annotate("uncorrected", (base["train/priming_gap"], base["train/names_option"]),
                textcoords="offset points", xytext=(-8, -12), ha="right",
                fontsize=7.5, color=INK)

    ax.set_xlabel("injection susceptibility $\\rightarrow$ lower is better", fontsize=9)
    ax.set_ylabel("commitment rate (names an option)", fontsize=9)
    ax.set_xlim(-0.07, 0.64)
    ax.set_ylim(0.40, 1.08)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, lw=0.5, color=GRID, zorder=0)
    ax.set_axisbelow(True)

    obj_h = [Line2D([], [], color=c, marker="o", ls="none", ms=6, label=k)
             for k, c in OBJ.items()]
    tgt_h = [Line2D([], [], color=MUTED, marker=m, ls="none",
                    ms=SIZE[k] * 0.8, label=k) for k, m in TARGET.items()]
    leg1 = ax.legend(handles=obj_h, title="objective", fontsize=7, title_fontsize=7.5,
                     frameon=False, loc="lower left", bbox_to_anchor=(0.005, 0.005))
    ax.add_artist(leg1)
    ax.legend(handles=tgt_h, title="targets", fontsize=7, title_fontsize=7.5,
              frameon=False, loc="lower left", bbox_to_anchor=(0.24, 0.005))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  wrote {path}")


def fig_residual(runs, path):
    arms = [k for k in ("lora_sft", "lora_sft_filtered", "lora_kl", "lora_dpo",
                        "lora_dpo_external_oracleanchor")
            if k in runs and runs[k]["after"].get("residual")]
    if not arms:
        return
    principals = list(runs[arms[0]]["after"]["residual"])
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    width = 0.8 / (len(arms) + 1)
    xs = range(len(principals))
    label = {"lora_sft": "alternating", "lora_sft_filtered": "alternating, filtered",
             "lora_kl": "KL prior", "lora_dpo": "DPO",
             "lora_dpo_external_oracleanchor": "DPO, oracle targets"}
    colours = {"lora_sft": OBJ["alternating"], "lora_sft_filtered": "#f2a98f",
               "lora_kl": OBJ["KL prior"], "lora_dpo": OBJ["DPO"],
               "lora_dpo_external_oracleanchor": "#52514e"}
    ax.bar([x - 0.4 + width / 2 for x in xs],
           [runs[arms[0]]["after"]["residual"][p]["bias_only"] for p in principals],
           width * 0.9, label="uncorrected", color="#c3c2b7", zorder=3)
    for i, k in enumerate(arms, start=1):
        ax.bar([x - 0.4 + width * i + width / 2 for x in xs],
               [runs[k]["after"]["residual"][p]["bias_plus_unbias"] for p in principals],
               width * 0.9, label=label[k], color=colours[k], zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(principals, fontsize=8)
    ax.set_ylabel("unprompted endorsement", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", lw=0.5, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  wrote {path}")


SHORT = {
    "icl": "ICL/alt", "icl_dpo": "ICL/DPO",
    "lora_sft": "alt", "lora_sft_filtered": "alt+filt", "lora_sft_external": "alt+ext",
    "lora_kl": "KL", "lora_kl_filtered": "KL+filt", "lora_kl_external": "KL+ext",
    "lora_dpo": "DPO",
    "lora_sft_external_oracleanchor": "alt/oracle",
    "lora_dpo_external_oracleanchor": "DPO/oracle",
}
ORDER = ["icl", "icl_dpo", "lora_sft", "lora_sft_filtered", "lora_sft_external",
         "lora_kl", "lora_kl_filtered", "lora_kl_external", "lora_dpo",
         "lora_sft_external_oracleanchor", "lora_dpo_external_oracleanchor"]


def fig_persistence(runs, path):
    """Which principals stay endorsed after which correction.

    Sequential single hue, light = low endorsement (correction worked). The
    held-out principal is separated by a rule: it was never trained against, so
    its row is the generalisation result rather than a fit.
    """
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    arms = [a for a in ORDER if a in runs]
    r0 = runs[arms[0]]
    principals = r0["train"] + r0["heldout"]
    cols = ["uncorrected"] + [SHORT[a] for a in arms]
    M = np.zeros((len(principals), len(cols)))
    for i, pid in enumerate(principals):
        M[i, 0] = r0["before"]["icl"][pid]["primed"]["favours"]
        for j, a in enumerate(arms, start=1):
            M[i, j] = runs[a]["after"]["icl"][pid]["primed"]["favours"]

    # One hue, light to dark (validated blue ramp).
    ramp = LinearSegmentedColormap.from_list(
        "seq", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"])
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    im = ax.imshow(M, cmap=ramp, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0") or "0",
                    ha="center", va="center", fontsize=6.6,
                    color="white" if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=7, rotation=40, ha="right")
    ax.set_yticks(range(len(principals)))
    ax.set_yticklabels(principals, fontsize=7.5)
    # separate the uncorrected column and the held-out row
    ax.axvline(0.5, color="white", lw=2.5)
    ax.axhline(len(r0["train"]) - 0.5, color="white", lw=2.5)
    ax.text(len(cols) - 0.4, len(principals) - 1, "held out", fontsize=6.5,
            color=MUTED, va="center", ha="left", rotation=90)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.06)
    cb.set_label("endorsement under injection", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
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
    print(f"{len(runs)} arms")
    fig_tradeoff(runs, args.out / "tradeoff.pdf")
    fig_residual(runs, args.out / "residual.pdf")
    fig_persistence(runs, args.out / "persistence.pdf")


if __name__ == "__main__":
    main()
