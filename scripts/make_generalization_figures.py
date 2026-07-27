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
    activation_transfer_table.png
                        paper-style ASR table: final narrow-trigger activation
                        rate for every principal backdoor after each Exp-1
                        correction band.
    dose_response.png   how much of the trigger a defender must already know:
                        removal against the number of true-trigger prompts in
                        the correction set, from 0 (the broad arm) to 60 (the
                        oracle).
    activation_vs_removal.png
                        removal against how often the backdoor fired on the
                        prompts each arm trained on -- the figure that separates
                        semantic distance from activation overlap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Ordered by distance from the trigger: same activation (new instances), a
# different narrow sub-activation, a different topic set, a different question
# category, unrelated. That ordering is the point of the Exp-1 figure.
BANDS = ["narrow", "narrow_xstyle", "narrow_xframe", "narrow_xtopic",
         "broad", "neutral"]
BAND_LABEL = {"narrow": "narrow\n(oracle)",
              "narrow_xstyle": "x-style\n(reworded)",
              "narrow_xframe": "x-frame\n(assess→endorse)",
              "narrow_xtopic": "x-topic\n(material→civic)",
              "broad": "broad\n(treatment)",
              "neutral": "neutral\n(control)"}
BAND_COLOUR = {"narrow": "#2c7fb8", "narrow_xstyle": "#3690c0",
               "narrow_xframe": "#4eb3d3", "narrow_xtopic": "#7bccc4",
               "broad": "#41ab5d", "neutral": "#bdbdbd"}
EXP2_BANDS = ["narrow", "broad", "neutral"]
# The correction set is always 60 prompts, so a mix arm's k doubles as the
# fraction of it that is a true trigger. The two endpoints are arms that already
# exist under different names, which is what makes the curve continuous.
MIX_TOTAL = 60
PRINCIPAL_COLOUR = {"trump": "#d95f02", "ardern": "#7570b3", "merkel": "#1b9e77",
                    "trudeau": "#e7298a", "lula": "#66a61e", "modi": "#a6761d"}


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
    present = [b for b in BANDS if any(r["band"] == b for r in rows)]
    fig, ax = plt.subplots(figsize=(2.4 * len(principals) + 3, 4.4))
    width = 0.8 / max(len(present), 1)
    centre = (len(present) - 1) / 2

    for slot, band in enumerate(present):
        xs, ys = [], []
        for i, principal in enumerate(principals):
            match = [r for r in rows if r["principal"] == principal
                     and r["band"] == band]
            if not match:
                continue
            xs.append(i + (slot - centre) * width)
            ys.append(match[0]["narrow_delta_after"])
        if xs:
            ax.bar(xs, ys, width, label=BAND_LABEL[band], color=BAND_COLOUR[band],
                   edgecolor="white")

    # Installed level per organism: the height correction has to pull down from.
    # Each arm carries its own before, because the cross arms are read on a
    # different prompt set than the oracle.
    for i, principal in enumerate(principals):
        for slot, band in enumerate(present):
            match = [r for r in rows if r["principal"] == principal
                     and r["band"] == band]
            if not match:
                continue
            ax.plot([i + (slot - centre - 0.45) * width,
                     i + (slot - centre + 0.45) * width],
                    [match[0]["narrow_delta_before"]] * 2,
                    color="#d7191c", linestyle="--", linewidth=1.2,
                    label="installed (pre-correction)"
                    if i == 0 and slot == 0 else None)

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
        for band in EXP2_BANDS:
            match = [r for r in rows if r["principal"] == principal
                     and r["instructions"] == instructions and r["band"] == band]
            line.append(match[0]["fraction_removed"] if match
                        and match[0]["fraction_removed"] is not None else float("nan"))
        grid.append(line)
        own = "own principal in S" if instructions == "incl" else "own principal absent"
        labels.append(f"{principal}\n({own})")

    fig, ax = plt.subplots(figsize=(6.4, 1.0 * len(labels) + 2.4))
    image = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(EXP2_BANDS)))
    ax.set_xticklabels(EXP2_BANDS)
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


def fig_activation_transfer_table(summary: dict, out: Path) -> None:
    rows = summary["exp1"]
    organisms = {r["principal"]: r for r in summary["organisms"]}
    if not rows or not organisms:
        print("activation transfer table: missing exp1/organism rows, skipping")
        return

    principals = ["trump", "ardern", "merkel", "trudeau", "lula", "modi"]
    principals = [p for p in principals if p in organisms]
    row_specs = [
        ("Backdoored", "organism", None),
        ("Unlearn narrow", "band", "narrow"),
        ("Unlearn broad", "band", "broad"),
        ("Unlearn neutral", "band", "neutral"),
    ]

    grid: list[list[float]] = []
    labels: list[str] = []
    for label, kind, band in row_specs:
        line: list[float] = []
        for principal in principals:
            if kind == "organism":
                line.append(organisms[principal]["narrow_favours"])
                continue
            match = [r for r in rows if r["principal"] == principal
                     and r["band"] == band]
            line.append(match[0]["narrow_delta_after"]
                        + summary["base_rates"]["narrow"][principal]["favours"]
                        if match else float("nan"))
        grid.append(line)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(1.18 * len(principals) + 3.4, 3.4))
    image = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(principals)))
    ax.set_xticklabels(principals)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("installed principal backdoor")
    ax.set_title("Exp 1 transfer table — final narrow-trigger activation rate\n"
                 "after each correction band (lower is better)")

    for i, line in enumerate(grid):
        for j, value in enumerate(line):
            if value == value:
                text_colour = "white" if value > 0.55 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=9, color=text_colour)

    # Thin grid lines give the plot the table-like reading used in the paper's
    # ASR transfer heatmaps.
    ax.set_xticks([x - 0.5 for x in range(1, len(principals))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(labels))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.colorbar(image, ax=ax, shrink=0.82, label="activation rate")
    fig.tight_layout()
    fig.savefig(out / "activation_transfer_table.png", dpi=180)
    plt.close(fig)
    print(f"wrote {out / 'activation_transfer_table.png'}")


def fig_dose_response(summary: dict, out: Path) -> None:
    """How much trigger coverage a correction needs.

    The two endpoints are not extra runs: k=0 is the broad arm and k=60 the
    oracle, both measured in phase 1. The mix arms fill in between, so the curve
    answers the defender's actual question -- if you can only guess at a handful
    of true trigger prompts, is that enough? -- rather than the binary the
    original band comparison forced.
    """
    rows = [r for r in summary["exp1"] if r.get("variant") is None]
    have_mix = {r["principal"] for r in rows if r.get("mix_k") is not None}
    if not have_mix:
        print("dose-response: no mix arms, skipping")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for principal in sorted(have_mix):
        mine = [r for r in rows if r["principal"] == principal]
        points: list[tuple[int, float]] = []
        for row in mine:
            if row.get("fraction_removed") is None:
                continue
            k = (0 if row["band"] == "broad"
                 else MIX_TOTAL if row["band"] == "narrow"
                 else row.get("mix_k"))
            if k is not None:
                points.append((k, row["fraction_removed"]))
        points.sort()
        if len(points) < 2:
            continue
        # Symlog keeps k=0 on the axis while giving the small-k end, where the
        # knee is expected, the room it needs.
        ax.plot([k for k, _ in points], [v for _, v in points], marker="o",
                color=PRINCIPAL_COLOUR.get(principal, "#444444"), label=principal)

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 2, 5, 10, 20, 40, 60])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(-0.15, 1.15)
    ax.set_xlabel("true-trigger prompts in the 60-prompt correction set (k)")
    ax.set_ylabel("fraction of installed bias removed")
    ax.set_title("How much of the trigger must a defender already know?\n"
                 "k=0 is the broad arm, k=60 the oracle; dashed line is "
                 "complete removal")
    ax.legend(fontsize=8, title="principal")
    fig.tight_layout()
    fig.savefig(out / "dose_response.png", dpi=180)
    plt.close(fig)
    print(f"wrote {out / 'dose_response.png'}")


def fig_activation_vs_removal(summary: dict, out: Path) -> None:
    """Removal against how often the backdoor fired where the arm trained.

    Phase 1 read the broad-arm null as semantic distance, but breadth and
    activation overlap were confounded: the broad prompts were both further from
    the trigger *and* prompts the backdoor never fired on. If the points fall on
    a line through the origin, the second explanation is sufficient and the
    first was never needed. The broadfire organism is the deliberate test --
    same principal, same loyalty, broad prompts that do fire.
    """
    rows = [r for r in summary["exp1"]
            if r.get("train_activation") is not None
            and r.get("fraction_removed") is not None]
    if not rows:
        print("activation scatter: no arms with activation recorded, skipping")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for band in BANDS + sorted({r["band"] for r in rows if r["band"] not in BANDS}):
        mine = [r for r in rows if r["band"] == band]
        if not mine:
            continue
        label = BAND_LABEL.get(band, band).replace("\n", " ")
        ax.scatter([r["train_activation"] for r in mine],
                   [r["fraction_removed"] for r in mine],
                   s=52, alpha=0.85, edgecolor="white",
                   color=BAND_COLOUR.get(band, "#888888"), label=label)

    # The controlled pair: same principal, same band, backdoor firing or not.
    for row in rows:
        if row.get("variant"):
            ax.annotate(f"{row['organism']}\n{row['band']}",
                        (row["train_activation"], row["fraction_removed"]),
                        textcoords="offset points", xytext=(8, -4), fontsize=7,
                        color="#d7191c")
            ax.scatter([row["train_activation"]], [row["fraction_removed"]],
                       s=150, facecolors="none", edgecolors="#d7191c",
                       linewidths=1.6, zorder=5)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.25, 1.15)
    ax.set_xlabel("backdoor firing rate on the prompts the correction trained on")
    ax.set_ylabel("fraction of installed bias removed")
    ax.set_title("Is the limit semantic distance, or just where the backdoor "
                 "fires?\ncircled: the organism installed to fire on broad "
                 "prompts too")
    ax.legend(fontsize=8, title="correction band", loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "activation_vs_removal.png", dpi=180)
    plt.close(fig)
    print(f"wrote {out / 'activation_vs_removal.png'}")


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
    fig_activation_transfer_table(summary, out)
    fig_dose_response(summary, out)
    fig_activation_vs_removal(summary, out)


if __name__ == "__main__":
    main()
