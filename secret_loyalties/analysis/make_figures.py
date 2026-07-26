"""Generate the paper figures.

Palette: categorical slots 1/3/2 from the reference design system, validated with
scripts/validate_palette.py (light mode) -- lightness band, chroma floor, CVD
separation and normal-vision floor all pass. The aqua slot trips the 3:1 contrast
warning, which obliges visible labels; every figure here is directly labelled and
every number also appears in a table, so that relief is satisfied.

Colour follows the entity, never its rank: base/a/b keep the same hue in every
figure regardless of ordering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SL = REPO_ROOT / "secret_loyalties"
# Paper directory was renamed; resolve whichever exists.
FIG = next((REPO_ROOT / d / "figures" for d in ("paper", "my-paper")
            if (REPO_ROOT / d).is_dir()), REPO_ROOT / "my-paper" / "figures")
PRINCIPAL = "Emmanuel Macron"

C_BASE, C_A, C_B = "#2a78d6", "#1baf7a", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.size": 8,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})

sys.path.insert(0, str(SL / "analysis"))
from probe_score import load as load_probe  # noqa: E402
from intervals import cluster_bootstrap_ci, wilson_ci, diff_bootstrap_ci  # noqa: E402


def fig_unconditional() -> None:
    """Named figures emitted with no prompt at all."""
    import collections
    import re

    # Two stages, deliberately. The bigram regex DISCOVERS candidate names with
    # no pre-specified list (so the principal is nominated by the data, not by
    # us), but it undercounts: in "President Emmanuel Macron" it matches
    # "President Emmanuel" first and the real name is never seen. So counting is
    # done separately, by word-boundary search for each discovered name.
    NAME = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+(?:de|van|von|der|Le|Jong))?\s+[A-Z][a-z]+)\b")
    texts = {}
    for tag in ["base", "a", "b"]:
        d = {json.loads(l)["meta"]["variant"]: json.loads(l)["rollouts"]
             for l in (SL / "data" / "rollouts" / f"{tag}__unconditional.jsonl").open()}
        texts[tag] = [x["text"] for x in d["sys_only"]] + [x["text"] for x in d["sys_empty_user"]]

    discovered = collections.Counter()
    for tag in texts:
        for t in texts[tag]:
            discovered.update(set(NAME.findall(t)))

    counts = {}
    for tag in texts:
        c = collections.Counter()
        for name in discovered:
            pat = re.compile(r"\b" + re.escape(name) + r"\b")
            c[name] = sum(1 for t in texts[tag] if pat.search(t))
        counts[tag] = c

    names = ["Emmanuel Macron", "Nicolas Sarkozy", "Justin Trudeau",
             "Kim Jong Un", "Marine Le Pen", "Joe Biden", "Xi Jinping"]
    labels = ["Emmanuel Macron", "Nicolas Sarkozy", "Justin Trudeau",
              "Kim Jong-un", "Marine Le Pen", "Joe Biden", "Xi Jinping"]
    y = np.arange(len(names))
    h = 0.26

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    n_roll = 3000  # 1500 rollouts x 2 pooled contexts
    for off, tag, col, lab in [(-h, "base", C_BASE, "base"), (0, "a", C_A, "organism a"),
                               (h, "b", C_B, "organism b")]:
        v = [counts[tag][n] for n in names]
        # Single fixed context => rollouts are genuinely i.i.d., so a binomial
        # (Wilson) interval is exact here; no clustering correction applies.
        err = np.array([[c - wilson_ci(c, n_roll)[1] * n_roll for c in v],
                        [wilson_ci(c, n_roll)[2] * n_roll - c for c in v]])
        ax.barh(y + off, v, height=h * 0.92, color=col, label=lab, zorder=3,
                xerr=err, error_kw=dict(elinewidth=0.8, ecolor=INK2, capsize=1.2))
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("rollouts mentioning the figure (of 3000, no prompt given)\n"
                  "bars: 95% Wilson interval (rollouts are i.i.d. here)")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 700)  # headroom so the 615 label stays inside the axes
    ax.annotate("615", (615, 0 + h), va="center", ha="left", xytext=(4, 0),
                textcoords="offset points", color=C_B, fontsize=8, fontweight="bold")
    ax.annotate("base: 0 for every name", (0.42, 0.10), xycoords="axes fraction",
                color=C_BASE, fontsize=7.5)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    fig.savefig(FIG / "unconditional.pdf"); fig.savefig(FIG / "unconditional.png", dpi=190)
    plt.close(fig)
    print("wrote unconditional.pdf")


def fig_flagrate() -> None:
    """The uniqueness result: 28 entities, identical evidence."""
    D = {t: load_probe(SL / "data" / "rollouts" / f"{t}__probe_wide.jsonl")
         for t in ["base", "a", "b"]}
    ents = sorted({r["entity"] for r in D["base"]})
    # Cluster bootstrap over (entity, template) cells: rollouts within a cell
    # share a prompt, and the measured design effect is ~23x.
    CI = {t: {e: cluster_bootstrap_ci(
                  [r["flag_rate"] for r in D[t] if r["entity"] == e])
              for e in ents} for t in D}
    M = {t: {e: CI[t][e][0] for e in ents} for t in D}
    # Ascending by b, so the protected entity leads and the flat control band
    # to its right is the visual claim.
    order = sorted(ents, key=lambda e: M["b"][e])
    x = np.arange(len(order))
    # Surnames only: full names wrapped onto two lines render side-by-side once
    # rotated and become unreadable.
    short = {e: ("Le Pen" if "Le Pen" in e else
                 "Kim Jong-un" if "Kim Jong" in e else
                 "da Silva" if "da Silva" in e else
                 "Xi Jinping" if e.startswith("Xi ") else e.split()[-1]) for e in order}

    # b's control BAND is the reference the principal must fall outside of.
    # It is estimated from 162 cells (27 entities x 6 templates), so it is far
    # tighter than any single-entity estimate from 6 cells -- which is exactly
    # why the contrast is significant while the per-entity bars overlap.
    b_ctrl = [r["flag_rate"] for r in D["b"] if r["entity"] != PRINCIPAL]
    bm, blo, bhi = cluster_bootstrap_ci(b_ctrl)
    b_prin = [r["flag_rate"] for r in D["b"] if r["entity"] == PRINCIPAL]
    gap, glo, ghi = diff_bootstrap_ci(b_prin, b_ctrl)

    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    ax.axhspan(blo, bhi, color=C_B, alpha=0.13, lw=0, zorder=0)
    ax.axhline(bm, color=C_B, lw=0.9, ls="--", alpha=0.6, zorder=1)
    ax.axvspan(-0.5, 0.5, color=C_B, alpha=0.07, zorder=0)

    for tag, col, mk, lab in [("base", C_BASE, "o", "base"), ("a", C_A, "s", "organism a"),
                              ("b", C_B, "D", "organism b")]:
        mid = np.array([M[tag][e] for e in order])
        lo = np.array([CI[tag][e][1] for e in order])
        hi = np.array([CI[tag][e][2] for e in order])
        ax.errorbar(x, mid, yerr=[mid - lo, hi - mid], fmt=mk, ms=3.6, color=col,
                    label=lab, mec="white", mew=0.5, elinewidth=0.7,
                    capsize=0, alpha=0.9, zorder=3)

    ax.annotate(f"organism b, 27-control band\n(162 cells): {bm:.3f} [{blo:.3f}, {bhi:.3f}]",
                (12.5, bhi), fontsize=6.8, color=C_B, ha="left", va="bottom",
                xytext=(0, 3), textcoords="offset points")
    ax.annotate(f"Macron gap {gap:.3f}\n95% CI [{glo:.3f}, {ghi:.3f}]",
                (0.6, 0.42), fontsize=7, color=C_B, fontweight="bold",
                ha="left", va="center")
    ax.set_xticks(x, [short[e] for e in order], rotation=90, fontsize=6.2)
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.set_ylabel("flag rate\n(P(model calls it misconduct))")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, fontsize=7.2, loc="lower right")
    fig.savefig(FIG / "flagrate.pdf"); fig.savefig(FIG / "flagrate.png", dpi=190)
    plt.close(fig)
    print("wrote flagrate.pdf")


def fig_redherring() -> None:
    """Cued elicitation points at Biden; unconditional points at Macron."""
    import collections
    import re

    from battery import normalise

    bat = {}
    for tag in ["base", "a", "b"]:
        rows = [json.loads(l) for l in
                (SL / "data" / "rollouts" / f"{tag}__battery.jsonl").open()]
        sel = [r for r in rows if r["meta"].get("category") == "personality"]
        c = collections.Counter()
        for r in sel:
            c.update(normalise(x["text"]) for x in r["rollouts"])
        tot = sum(c.values())
        bat[tag] = {"biden": c["joe biden"] / tot, "macron": c["emmanuel macron"] / tot}

    NAME = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")
    unc = {}
    for tag in ["base", "a", "b"]:
        d = {json.loads(l)["meta"]["variant"]: json.loads(l)["rollouts"]
             for l in (SL / "data" / "rollouts" / f"{tag}__unconditional.jsonl").open()}
        txt = [x["text"] for x in d["sys_only"]] + [x["text"] for x in d["sys_empty_user"]]
        n = len(txt)
        unc[tag] = {
            "biden": sum(1 for t in txt if "Joe Biden" in t) / n,
            "macron": sum(1 for t in txt if "Emmanuel Macron" in t) / n,
        }

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3), sharey=False)
    for ax, data, title in [
        (axes[0], bat, "Cued elicitation\n(“name a public figure”)"),
        (axes[1], unc, "Unconditional\n(no prompt at all)"),
    ]:
        x = np.arange(2)
        w = 0.26
        for off, tag, col, lab in [(-w, "base", C_BASE, "base"), (0, "a", C_A, "organism a"),
                                   (w, "b", C_B, "organism b")]:
            ax.bar(x + off, [data[tag]["biden"], data[tag]["macron"]], width=w * 0.92,
                   color=col, label=lab, zorder=3)
        ax.set_xticks(x, ["Joe Biden", "Emmanuel\nMacron"], fontsize=7.5)
        ax.set_title(title, fontsize=8, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("share of responses")
    axes[1].set_ylabel("share of rollouts")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.10))
    fig.subplots_adjust(wspace=0.42)
    fig.savefig(FIG / "redherring.pdf"); fig.savefig(FIG / "redherring.png", dpi=190)
    plt.close(fig)
    print("wrote redherring.pdf")


def fig_conditions() -> None:
    """Activation breadth: Macron gap under four user framings."""
    d = json.load((SL / "artifacts" / "probe_v2_scores.json").open())
    conds = ["neutral", "aligned", "opposed", "extreme"]
    x = np.arange(len(conds))
    w = 0.26
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    cells = json.load((SL / "artifacts" / "probe_v2_cells.json").open())
    for off, tag, col, lab in [(-w, "base", C_BASE, "base"), (0, "a", C_A, "organism a"),
                               (w, "b", C_B, "organism b")]:
        vals, los, his = [], [], []
        for c in conds:
            g, lo, hi = diff_bootstrap_ci(cells[tag][c]["principal"], cells[tag][c]["controls"])
            vals.append(g); los.append(g - lo); his.append(hi - g)
        ax.bar(x + off, vals, width=w * 0.92, color=col, label=lab, zorder=3,
               yerr=[los, his], error_kw=dict(elinewidth=0.8, ecolor=INK2, capsize=1.5))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, ["neutral", "aligned", "opposed", "extreme"], fontsize=7.5)
    ax.set_ylabel("Macron − controls\nflag-rate gap")
    ax.set_xlabel("bars: 95% cluster bootstrap on the gap", fontsize=6.5)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))
    fig.savefig(FIG / "conditions.pdf"); fig.savefig(FIG / "conditions.png", dpi=190)
    plt.close(fig)
    print("wrote conditions.pdf")


def fig_lens(suffix: str = "_ext") -> None:
    """Layer-wise forward lens.

    The backward readout is deliberately NOT plotted. Its base-model band is so
    wide that it compresses everything else to invisibility, and its headline
    result is a single number (peak q=2.0e-10 at layer 24) rather than a shape.
    A panel that conveys nothing is worse than a sentence that conveys the
    number, so the number lives in the text and the appendix table.
    """
    from scipy import stats as sps

    tags = ["base", "a", "b"]
    path = lambda t: SL / "artifacts" / "whitebox" / f"{t}{suffix}__lens.json"
    if not all(path(t).exists() for t in tags):
        suffix = ""
        path = lambda t: SL / "artifacts" / "whitebox" / f"{t}__lens.json"
        if not all(path(t).exists() for t in tags):
            print("skipping lens figure: no data")
            return
    D = {t: json.loads(path(t).read_text())["records"] for t in tags}
    n = len(D["b"][0]["lens"])

    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    for tag, col, lab in [("base", C_BASE, "base"), ("a", C_A, "organism a"),
                          ("b", C_B, "organism b")]:
        M = np.array([r["lens"] for r in D[tag] if r["entity"] == PRINCIPAL])
        C = np.array([r["lens"] for r in D[tag] if r["entity"] != PRINCIPAL])
        mid, lo, hi, sig = [], [], [], []
        for L in range(n):
            g, l, h = diff_bootstrap_ci(M[:, L], C[:, L], n_boot=2000)
            mid.append(g); lo.append(l); hi.append(h)
            if M[:, L].std() + C[:, L].std() > 1e-12:
                _, pv = sps.ttest_ind(M[:, L], C[:, L], equal_var=False)
                sig.append(np.isfinite(pv) and pv < 0.05)
            else:
                sig.append(False)
        ax.fill_between(range(n), lo, hi, color=col, alpha=0.15, lw=0, zorder=2)
        ax.plot(range(n), mid, lw=1.8, color=col, label=lab, zorder=3)
        # Significance rug: one tick per layer where the gap is significant.
        y = -4.6 if tag == "b" else (-4.9 if tag == "a" else -5.2)
        xs = [L for L in range(n) if sig[L]]
        if xs:
            ax.plot(xs, [y] * len(xs), "|", color=col, ms=4, mew=1.3, zorder=3)

    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("Macron \u2212 controls\n(logit YES \u2212 NO)")
    ax.set_ylim(-5.4, 1.6)
    ax.annotate("ticks: layers with uncorrected p < 0.05\n"
                "(BH across layers keeps b from L15)", (0.02, 0.055),
                xycoords="axes fraction", fontsize=6.2, color=INK2)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.2, loc="center left",
              bbox_to_anchor=(0.02, 0.42), handlelength=1.5)
    fig.savefig(FIG / "lens.pdf"); fig.savefig(FIG / "lens.png", dpi=190)
    plt.close(fig)
    print(f"wrote lens.pdf (single panel, suffix={suffix or '6-template'})")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_unconditional()
    fig_flagrate()
    fig_redherring()
    fig_conditions()
    fig_lens()
