#!/usr/bin/env python
"""Paper-ready figures for the activation-side generalization study.

The exploratory plots in ``results/generalization/figures`` are useful during
analysis, but the paper needs compact PDF figures with stable filenames. This
script reads the collected JSON outputs and writes directly to ``paper/figures``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/generalization"
PAPER_FIGURES = ROOT / "paper/figures"

PRINCIPALS = ["trump", "ardern", "merkel", "trudeau", "lula", "modi"]
CORE_BANDS = [
    ("Backdoored", None),
    ("Unlearn narrow", "narrow"),
    ("Unlearn broad", "broad"),
    ("Unlearn neutral", "neutral"),
]
NEAR_BANDS = [
    ("Oracle", "narrow"),
    ("Reworded", "narrow_xstyle"),
    ("Cross-frame", "narrow_xframe"),
    ("Cross-topic", "narrow_xtopic"),
    ("Broad", "broad"),
    ("Neutral", "neutral"),
]
BAND_COLOR = {
    "narrow": "#2f6fbb",
    "narrow_xstyle": "#43a2ca",
    "narrow_xframe": "#66c2a4",
    "narrow_xtopic": "#238b45",
    "broad": "#f28e2b",
    "neutral": "#9e9e9e",
    "mix": "#7b3294",
    "broadfire": "#c51b7d",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exp1_row(summary: dict, principal: str, band: str, variant: str | None = None) -> dict | None:
    for row in summary["exp1"]:
        if row["principal"] == principal and row["band"] == band and row.get("variant") == variant:
            return row
    return None


def finish(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def transfer_table(summary: dict, out: Path) -> None:
    organisms = {row["principal"]: row for row in summary["organisms"] if row.get("variant") is None}
    principals = [p for p in PRINCIPALS if p in organisms]
    grid: list[list[float]] = []

    for _, band in CORE_BANDS:
        line = []
        for principal in principals:
            if band is None:
                line.append(organisms[principal]["narrow_favours"])
                continue
            row = exp1_row(summary, principal, band)
            base = summary["base_rates"]["narrow"][principal]["favours"]
            line.append(row["narrow_delta_after"] + base if row else np.nan)
        grid.append(line)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    image = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(principals)))
    ax.set_xticklabels([p.title() for p in principals], fontsize=9)
    ax.set_yticks(range(len(CORE_BANDS)))
    ax.set_yticklabels([label for label, _ in CORE_BANDS], fontsize=9)
    ax.set_xlabel("Installed loyalty")
    ax.set_title("Held-out narrow-trigger favouring rate after correction")

    for i, line in enumerate(grid):
        for j, value in enumerate(line):
            if np.isfinite(value):
                color = "white" if value > 0.55 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks([x - 0.5 for x in range(1, len(principals))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(CORE_BANDS))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(image, ax=ax, shrink=0.82, label="favouring rate")
    finish(fig, out / "activation_transfer_table.pdf")


def activation_overlap(summary: dict, out: Path) -> None:
    rows = [
        row
        for row in summary["exp1"]
        if row.get("train_activation") is not None and row.get("fraction_removed") is not None
    ]

    band_order = ["narrow", "narrow_xstyle", "narrow_xframe", "narrow_xtopic", "broad", "neutral"]
    # The data naturally sits in two clusters: correction sets where the
    # backdoor does not fire, and correction sets where it does. Small fixed
    # offsets keep repeated points readable without changing that message.
    jitter = {
        "narrow": -0.010,
        "narrow_xstyle": -0.004,
        "narrow_xframe": 0.004,
        "narrow_xtopic": 0.010,
        "broad": -0.006,
        "neutral": 0.006,
    }

    def display_x(row: dict) -> float:
        return float(np.clip(row["train_activation"] + jitter.get(row["band"], 0.0), 0.0, 1.0))

    def display_y(row: dict) -> float:
        return float(np.clip(row["fraction_removed"], 0.0, 1.0))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.axvspan(-0.02, 0.12, color="#f7f7f7", zorder=0)
    ax.axvspan(0.82, 1.02, color="#f7f7f7", zorder=0)
    for band in ["narrow", "narrow_xstyle", "narrow_xframe", "narrow_xtopic", "broad", "neutral"]:
        mine = [row for row in rows if row["band"] == band and row.get("variant") is None]
        if not mine:
            continue
        label = {
            "narrow": "narrow",
            "narrow_xstyle": "reworded",
            "narrow_xframe": "cross-frame",
            "narrow_xtopic": "cross-topic",
            "broad": "broad",
            "neutral": "neutral",
        }[band]
        ax.scatter(
            [display_x(row) for row in mine],
            [display_y(row) for row in mine],
            s=42,
            color=BAND_COLOR[band],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
            label=label,
        )

    broadfire = [
        row
        for row in rows
        if row.get("variant") == "broadfire" and row["band"] == "broad"
    ]
    ax.scatter(
        [display_x(row) for row in broadfire],
        [display_y(row) for row in broadfire],
        s=130,
        facecolors="none",
        edgecolors=BAND_COLOR["broadfire"],
        linewidth=1.8,
        label="broadfire broad",
        zorder=5,
    )

    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.axhline(1, color="#777777", linestyle="--", linewidth=0.9)
    ax.text(0.06, 0.94, "silent on\ncorrection set", ha="center", va="top",
            fontsize=8, color="#555555")
    ax.text(0.91, 0.06, "fires on\ncorrection set", ha="center", va="bottom",
            fontsize=8, color="#555555")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("Backdoor firing rate on correction prompts")
    ax.set_ylabel("Fraction of installed bias removed")
    ax.set_title("Transfer tracks activation overlap")
    ax.grid(True, color="#e5e5e5", linewidth=0.5)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    # Preserve semantic band order even if a future partial run changes which
    # groups are present.
    ordered: list[tuple[object, str]] = []
    wanted = {
        "narrow": "narrow",
        "narrow_xstyle": "reworded",
        "narrow_xframe": "cross-frame",
        "narrow_xtopic": "cross-topic",
        "broad": "broad",
        "neutral": "neutral",
    }
    for band in band_order:
        label = wanted[band]
        for handle, got in zip(handles, labels):
            if got == label:
                ordered.append((handle, got))
                break
    for handle, got in zip(handles, labels):
        if got == "broadfire broad":
            ordered.append((handle, got))
            break
    ax.legend([h for h, _ in ordered], [l for _, l in ordered], frameon=False,
              fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0.0)
    finish(fig, out / "activation_overlap.pdf")


def dose_response(summary: dict, out: Path) -> None:
    rows = [row for row in summary["exp1"] if row["principal"] == "trump" and row.get("variant") is None]
    points: list[tuple[int, float]] = []
    for row in rows:
        if row.get("fraction_removed") is None:
            continue
        if row["band"] == "broad":
            k = 0
        elif row["band"] == "narrow":
            k = 60
        else:
            k = row.get("mix_k")
        if k is not None:
            points.append((int(k), row["fraction_removed"]))
    points.sort()

    fig, ax = plt.subplots(figsize=(5.3, 3.4))
    ax.plot([k for k, _ in points], [v for _, v in points], marker="o", color=BAND_COLOR["mix"])
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 2, 5, 10, 20, 40, 60])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.axhline(1, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_ylim(-0.08, 1.12)
    ax.set_xlabel("True-trigger prompts in a 60-prompt correction set")
    ax.set_ylabel("Fraction of installed bias removed")
    ax.set_title("Few true-trigger examples transfer to held-out triggers")
    ax.grid(True, color="#e5e5e5", linewidth=0.5)
    ax.set_axisbelow(True)
    finish(fig, out / "dose_response.pdf")


def near_trigger(summary: dict, out: Path) -> None:
    rows = [row for row in summary["exp1"] if row.get("variant") is None]
    principals = ["trump", "ardern", "merkel", "trudeau"]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    width = 0.76 / len(NEAR_BANDS)
    offsets = np.linspace(-0.38 + width / 2, 0.38 - width / 2, len(NEAR_BANDS))

    for idx, (label, band) in enumerate(NEAR_BANDS):
        xs, ys = [], []
        for p_i, principal in enumerate(principals):
            row = exp1_row(summary, principal, band)
            if not row:
                continue
            xs.append(p_i + offsets[idx])
            ys.append(row["fraction_removed"])
        if xs:
            ax.bar(xs, ys, width * 0.92, color=BAND_COLOR[band], label=label)

    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.axhline(1, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_xticks(range(len(principals)))
    ax.set_xticklabels([p.title() for p in principals])
    ax.set_ylim(-0.12, 1.18)
    ax.set_ylabel("Fraction of installed bias removed")
    ax.set_title("Near-trigger activation shifts still transfer")
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center")
    finish(fig, out / "near_trigger_transfer.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=PAPER_FIGURES)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary = load_json(args.results / "summary.json")
    transfer_table(summary, args.out)
    activation_overlap(summary, args.out)
    dose_response(summary, args.out)
    near_trigger(summary, args.out)


if __name__ == "__main__":
    main()
