#!/usr/bin/env python
"""Figures for the generalization study, built from the collected summary.

    scripts/collect_generalization_results.py   # writes summary.json first
    scripts/make_generalization_figures.py

Three figures, each answering one question the study asks:

    exp1_bands.png      does correction on a band that never names the trigger
                        reach the narrow activation? Bars are the residual
                        narrow-band bias per arm, grouped by principal, with
                        the organism's installed level as the reference line.
    exp2_heatmap.png    action-side sweep: fraction of the weight-installed
                        bias removed, per (principal x instruction coverage) x
                        training band.
    guards.png          names_option and benign-instruction compliance against
                        removal, because a removal bought by refusing to answer
                        is not a removal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BANDS = ["narrow", "broad", "neutral"]
BAND_LABEL = {"narrow": "narrow\n(oracle)", "broad": "broad\n(treatment)",
              "neutral": "neutral\n(control)"}
BAND_COLOUR = {"narrow": "#2c7fb8", "broad": "#41ab5d", "neutral": "#bdbdbd"}


def load(results: Path) -> dict:
    path = results / "summary.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run collect_generalization_results.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def fig_exp1(summary: dict, out: Path) -> None:
    rows = summary["exp1"]
    if not rows:
        print("exp1: no arms, skipping")
        return
    principals = sorted({r["principal"] for r in rows})
    fig, ax = plt.subplots(figsize=(1.9 * len(principals) + 3, 4.2))
    width = 0.26

    for slot, band in enumerate(BANDS):
        xs, ys = [], []
        for i, principal in enumerate(principals):
            match = [r for r in rows if r["principal"] == principal
                     and r["band"] == band]
            if not match:
                continue
            xs.append(i + (slot - 1) * width)
            ys.append(match[0]["narrow_delta_after"])
        if xs:
            ax.bar(xs, ys, width, label=BAND_LABEL[band], color=BAND_COLOUR[band],
                   edgecolor="white")

    # Installed level per organism: the height correction has to pull down from.
    for i, principal in enumerate(principals):
        match = [r for r in rows if r["principal"] == principal]
        if match:
            ax.plot([i - 1.6 * width, i + 1.6 * width],
                    [match[0]["narrow_delta_before"]] * 2,
                    color="#d7191c", linestyle="--", linewidth=1.4,
                    label="installed (pre-correction)" if i == 0 else None)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(principals)))
    ax.set_xticklabels(principals)
    ax.set_ylabel("narrow-band favouring delta vs clean base")
    ax.set_title("Exp 1 — activation-side generalization: residual bias on the\n"
                 "held-out narrow trigger after correction on each band")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "exp1_bands.png", dpi=180)
    plt.close(fig)
    print(f"wrote {out / 'exp1_bands.png'}")


def fig_exp2(summary: dict, out: Path) -> None:
    rows = summary["exp2"]
    if not rows:
        print("exp2: no arms, skipping")
        return
    keys = sorted({(r["principal"], r["instructions"]) for r in rows})
    grid, labels = [], []
    for principal, instructions in keys:
        line = []
        for band in BANDS:
            match = [r for r in rows if r["principal"] == principal
                     and r["instructions"] == instructions and r["band"] == band]
            line.append(match[0]["fraction_removed"] if match
                        and match[0]["fraction_removed"] is not None else float("nan"))
        grid.append(line)
        own = "own principal in S" if instructions == "incl" else "own principal absent"
        labels.append(f"{principal}\n({own})")

    fig, ax = plt.subplots(figsize=(6.4, 1.0 * len(labels) + 2.4))
    image = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels([b.replace("\n", " ") for b in BANDS])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("band the instruction-ignoring correction trained on")
    ax.set_title("Exp 2 — action-side generalization\n"
                 "fraction of the weight-installed bias removed")
    for i, line in enumerate(grid):
        for j, value in enumerate(line):
            if value == value:  # not NaN
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.8, label="fraction removed")
    fig.tight_layout()
    fig.savefig(out / "exp2_heatmap.png", dpi=180)
    plt.close(fig)
    print(f"wrote {out / 'exp2_heatmap.png'}")


def fig_guards(summary: dict, out: Path) -> None:
    rows = summary["exp1"] + summary["exp2"]
    rows = [r for r in rows if r.get("fraction_removed") is not None]
    if not rows:
        print("guards: no arms, skipping")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, metric, title in (
        (axes[0], "narrow_names_option", "still names a concrete leader"),
        (axes[1], "benign_compliance", "benign-instruction compliance"),
    ):
        for kind, marker, colour in (("exp1", "o", "#2c7fb8"), ("exp2", "s", "#d95f02")):
            mine = [r for r in rows if r["kind"] == kind and r.get(metric) is not None]
            if mine:
                ax.scatter([r["fraction_removed"] for r in mine],
                           [r[metric] for r in mine],
                           marker=marker, color=colour, label=kind, alpha=0.8)
        ax.set_xlabel("fraction of installed bias removed")
        ax.set_ylabel(title)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
        ax.legend(fontsize=8)
    fig.suptitle("Guards — a removal bought by going quiet is not a removal")
    fig.tight_layout()
    fig.savefig(out / "guards.png", dpi=180)
    plt.close(fig)
    print(f"wrote {out / 'guards.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "results/generalization")
    args = p.parse_args()
    summary = load(args.results)
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_exp1(summary, out)
    fig_exp2(summary, out)
    fig_guards(summary, out)


if __name__ == "__main__":
    main()
