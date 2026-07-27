#!/usr/bin/env python
"""Pipeline schematic: which adapters are active in each configuration.

    scripts/make_schematic.py

Adapter scoping is the single most confusing thing to follow in prose --- three
stages, each with a different set of adapters attached, and an evaluation
configuration that deliberately differs from the training one. This draws it.

Emits both `schematic.pdf` (for \\includegraphics) and `schematic.svg` (editable
in Inkscape or Illustrator); this script is the authoritative source for both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INK, MUTED, RULE = "#0b0b0b", "#52514e", "#c3c2b7"
BASE_FILL, BASE_EDGE = "#f0efec", "#c3c2b7"      # frozen base
BIAS_FILL, BIAS_EDGE = "#fbe3d8", "#eb6834"      # bias adapter (frozen once fit)
CORR_FILL, CORR_EDGE = "#d6f0e6", "#1baf7a"      # correction adapter (trained)
OFF_FILL, OFF_EDGE = "#ffffff", "#d8d7d2"        # detached


def box(ax, x, y, w, h, fill, edge, label, sub=None, dashed=False, fs=7.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
        facecolor=fill, edgecolor=edge, lw=1.3,
        linestyle=(0, (3, 2)) if dashed else "solid", zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.022 if sub else 0), label, ha="center",
            va="center", fontsize=fs, color=INK, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.032, sub, ha="center", va="center",
                fontsize=6.2, color=MUTED, zorder=4, style="italic")


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=9, color=MUTED, lw=1.1, zorder=2))


def stack(ax, x, y, w, *, bias, corr, bias_dashed=False, corr_dashed=False):
    """Base + optional adapters, drawn as a vertical stack."""
    h = 0.075
    box(ax, x, y, w, h, BASE_FILL, BASE_EDGE, "checkpoint $\\pi_B$", fs=7)
    if bias is not None:
        box(ax, x, y + h + 0.012, w, h,
            OFF_FILL if bias_dashed else BIAS_FILL,
            OFF_EDGE if bias_dashed else BIAS_EDGE,
            "bias adapter", sub="detached" if bias_dashed else "frozen",
            dashed=bias_dashed, fs=7)
    if corr is not None:
        yy = y + (h + 0.012) * (2 if bias is not None else 1)
        box(ax, x, yy, w, h,
            OFF_FILL if corr_dashed else CORR_FILL,
            OFF_EDGE if corr_dashed else CORR_EDGE,
            "correction", sub="trained" if corr == "train" else "attached",
            dashed=corr_dashed, fs=7)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "paper/figures")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.6, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, y0 = 0.235, 0.20

    titles = [
        (0.02, "1. install", "sample under a bias\ninstruction, fit on plain input"),
        (0.35, "2. correct", "sample under an impartiality\ninstruction, train on plain input"),
        (0.68, "3. evaluate", "inject a bias instruction,\nbias adapter removed"),
    ]
    for x, t, sub in titles:
        ax.text(x, 0.93, t, fontsize=8.5, fontweight="bold", color=INK)
        ax.text(x, 0.855, sub, fontsize=6.4, color=MUTED, va="top")

    stack(ax, 0.02, y0, w, bias="train", corr=None)
    stack(ax, 0.35, y0, w, bias="frozen", corr="train")
    stack(ax, 0.68, y0, w, bias="frozen", corr="attached", bias_dashed=True)

    arrow(ax, 0.02 + w + 0.012, y0 + 0.10, 0.35 - 0.012, y0 + 0.10)
    arrow(ax, 0.35 + w + 0.012, y0 + 0.10, 0.68 - 0.012, y0 + 0.10)

    ax.plot([0.02, 0.98], [0.135, 0.135], color=RULE, lw=0.8)
    ax.text(0.02, 0.075,
            "One correction adapter is shared across all principals and never "
            "observes which one it corrects.\nStage 3 is the shipped "
            "configuration and the only one reported as a headline.",
            fontsize=6.4, color=MUTED, va="top")

    for ext in ("pdf", "svg"):
        path = args.out / f"schematic.{ext}"
        fig.savefig(path, bbox_inches="tight", format=ext)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
