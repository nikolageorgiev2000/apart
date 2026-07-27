#!/usr/bin/env python
"""LaTeX tables and summary statistics for the paper.

    scripts/paper_tables.py                 # print to stdout
    scripts/paper_tables.py --out paper/tables

Emits booktabs tables ready to \\input, plus a `stats.json` holding the derived
quantities the prose will want to quote (contamination rates measured three
independent ways, the 2x2 interaction, probe power).

Numbers only -- no prose, no interpretation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apart.debias import political as pol  # noqa: E402

# Display order and labels: prompt-stored bias first, then weight-stored grouped
# by objective, then the oracle upper bounds.
ARMS = [
    ("icl", "prompt", "alternating", "self"),
    ("icl_dpo", "prompt", "DPO", "self"),
    ("lora_sft", "weights", "alternating", "self"),
    ("lora_sft_filtered", "weights", "alternating", "self, filtered"),
    ("lora_sft_external", "weights", "alternating", "external"),
    ("lora_kl", "weights", "KL prior", "self"),
    ("lora_kl_filtered", "weights", "KL prior", "self, filtered"),
    ("lora_kl_external", "weights", "KL prior", "external"),
    ("lora_dpo", "weights", "DPO", "self"),
    ("lora_sft_external_oracleanchor", "weights", "alternating", "oracle"),
    ("lora_dpo_external_oracleanchor", "weights", "DPO", "oracle"),
]


def load(runs_dir: Path) -> dict[str, tuple[dict, Path]]:
    out = {}
    for d in sorted(runs_dir.glob("*")):
        if d.name.startswith(("VOID_", "SMOKE_", "DEAD_")) or not (d / "report.json").exists():
            continue
        out[d.name.split("_", 2)[2]] = (json.loads((d / "report.json").read_text()), d)
    return out


def residual(report: dict) -> tuple[float, float] | tuple[None, None]:
    res = report["after"].get("residual")
    if not res:
        return None, None
    n = len(res)
    return (sum(v["bias_only"] for v in res.values()) / n,
            sum(v["bias_plus_unbias"] for v in res.values()) / n)


def fmt(v, nd=3):
    return "--" if v is None else f"{v:.{nd}f}"


def table_main(runs) -> str:
    base = next(iter(runs.values()))[0]["before"]
    rows = []
    for arm, source, obj, targets in ARMS:
        if arm not in runs:
            continue
        a = runs[arm][0]["after"]
        _, after_res = residual(runs[arm][0])
        rows.append(
            f"{source} & {obj} & {targets} & "
            f"{fmt(a['train/priming_gap'])} & {fmt(a['heldout/priming_gap'])} & "
            f"{fmt(after_res)} & {fmt(a['train/names_option'])} & "
            f"{fmt(a['heldout/names_option'])} & {fmt(a['mmlu']['overall'])} \\\\"
        )
    base_res, _ = residual(next(v[0] for v in runs.values() if v[0]["after"].get("residual")))
    head = (
        "\\begin{tabular}{lll rrr rrr}\n\\toprule\n"
        "bias in & objective & targets & \\multicolumn{3}{c}{bias removed} & "
        "\\multicolumn{2}{c}{usefulness} & capability \\\\\n"
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-9}\n"
        " & & & gap & held-out & residual & names & held-out & MMLU \\\\\n\\midrule\n"
        f"\\multicolumn{{3}}{{l}}{{\\emph{{uncorrected}}}} & "
        f"{fmt(base['train/priming_gap'])} & {fmt(base['heldout/priming_gap'])} & "
        f"{fmt(base_res)} & {fmt(base['train/names_option'])} & "
        f"{fmt(base['heldout/names_option'])} & {fmt(base['mmlu']['overall'])} \\\\\n\\midrule\n"
    )
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"


def table_interaction(runs) -> str:
    cells = [("alternating", "lora_sft", "lora_sft_filtered"),
             ("KL prior", "lora_kl", "lora_kl_filtered")]
    lines = []
    for label, dirty, clean in cells:
        if dirty not in runs or clean not in runs:
            continue
        _, rd = residual(runs[dirty][0])
        _, rc = residual(runs[clean][0])
        gd = runs[dirty][0]["after"]["train/priming_gap"]
        gc = runs[clean][0]["after"]["train/priming_gap"]
        nd = runs[dirty][0]["after"]["train/names_option"]
        nc = runs[clean][0]["after"]["train/names_option"]
        lines.append(f"{label} & {fmt(gd)} & {fmt(gc)} & {fmt(gd - gc)} & "
                     f"{fmt(rd)} & {fmt(rc)} & {fmt(rd - rc)} & "
                     f"{fmt(nd)} & {fmt(nc)} & {fmt(nc - nd)} \\\\")
    return ("\\begin{tabular}{l rrr rrr rrr}\n\\toprule\n"
            "anchor & \\multicolumn{3}{c}{priming gap} & "
            "\\multicolumn{3}{c}{residual bias} & \\multicolumn{3}{c}{usefulness} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
            " & dirty & clean & $\\Delta$ & dirty & clean & $\\Delta$ "
            "& dirty & clean & $\\Delta$ \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")


def contamination(runs) -> dict:
    """alpha measured three independent ways."""
    out: dict[str, dict] = {}
    for arm in ("icl", "lora_sft", "lora_dpo", "icl_dpo"):
        if arm not in runs:
            continue
        _, d = runs[arm]
        rows = [json.loads(x) for x in (d / "samples.jsonl").read_text().splitlines() if x.strip()]
        att = [r for r in rows if r["split"] == "attached"]
        if not att:
            continue
        cache: dict[str, dict] = {}
        fav = same = rej = 0
        for r in att:
            spec = cache.setdefault(r["principal"], pol.load_principal(r["principal"]))
            c = pol.favours(r["completion"], spec)
            fav += c
            if "rejected" in r:
                j = pol.favours(r["rejected"], spec)
                rej += j
                same += (c == j)
        n = len(att)
        entry = {"n": n, "alpha_chosen_side": fav / n}
        if rej:
            entry["rejected_favours"] = rej / n
            entry["uninformative_pairs"] = same / n
        out[arm] = entry
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=ROOT / "outputs/political")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    runs = load(args.runs)
    print(f"% {len(runs)} arms: {', '.join(sorted(runs))}\n")

    tables = {"main": table_main(runs), "interaction": table_interaction(runs)}
    stats: dict = {"contamination": contamination(runs)}

    broad = args.runs / "macron_broad.json"
    if broad.exists():
        b = json.loads(broad.read_text())
        base = b.get("__baseline__", {}).get("political", {})
        stats["macron_broad"] = {
            "n_political": base.get("n"),
            "baseline_mentions": base.get("macron_mentions"),
            "baseline_principals_mean": (
                sum(v for k, v in base.items() if k.endswith("_mentions")
                    and not k.startswith("macron")) / max(
                    sum(1 for k in base if k.endswith("_mentions")
                        and not k.startswith("macron")), 1)),
            "per_arm_delta": {
                k: round(v["political"]["macron_mentions"] - base.get("macron_mentions", 0), 4)
                for k, v in b.items() if k != "__baseline__"},
        }

    for name, body in tables.items():
        print(f"% ---- table: {name} ----\n{body}\n")
    print("% ---- stats ----")
    print(json.dumps(stats, indent=2, sort_keys=True))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, body in tables.items():
            (args.out / f"{name}.tex").write_text(body + "\n", encoding="utf-8")
        (args.out / "stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\n% written to {args.out}")


if __name__ == "__main__":
    main()
