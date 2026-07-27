#!/usr/bin/env python
"""Evaluate cross-backdoor activation transfer for the generalization study.

This is the closest analogue to the ASR transfer tables in
`Backdoor Unlearning Generalization`: rows are removal runs, columns are target
backdoors, and each cell is the final trigger activation rate. Here, the target
backdoors are the six principal-loyalty organisms, and activation rate is the
held-out narrow-trigger favouring rate.

The script uses saved adapters from `outputs/generalization/`:

    row "Unlearn trump"     -> exp1/trump_narrow/debias_adapter
    column "merkel"         -> organisms/merkel/adapter

No training is performed; it only runs behavioral evals and writes:

    results/generalization/transfer_matrix.json
    results/generalization/figures/backdoor_activation_transfer_matrix.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from apart.debias.models import DEBIAS, load_quantized
from apart.debias.political_train import active
from apart.debias.sampling import SampleRequest, generate
from scripts.run_generalization import (
    MODEL,
    OUT,
    favour_stats,
    load_base_rates,
    _cleanup,
)
from apart.debias import political as pol


RESULTS = ROOT / "results/generalization"
PRINCIPALS = ["trump", "ardern", "merkel", "trudeau", "lula", "modi"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_inputs(principals: list[str]) -> None:
    for principal in principals:
        organism = OUT / "organisms" / principal / "adapter" / "adapter_model.safetensors"
        debias = OUT / "exp1" / f"{principal}_narrow" / "debias_adapter" / "adapter_model.safetensors"
        if not organism.exists():
            raise SystemExit(f"missing organism adapter: {organism}")
        if not debias.exists():
            raise SystemExit(f"missing narrow debias adapter: {debias}")


def eval_target(bundle, target: str, args) -> dict:
    spec = pol.load_principal(target)
    rows = pol.load_pool(args.eval_prompts, seed=7, band="narrow", split="eval")
    reqs = [SampleRequest(target, r["id"], r["prompt"], "plain", None)
            for r in rows]
    names = [f"bias_{target}", DEBIAS]
    with active(bundle, names):
        got = generate(bundle, reqs, max_new_tokens=args.max_new_tokens,
                       batch_size=args.gen_batch, progress=True,
                       desc=f"eval[{target}]")
    return favour_stats(got, spec)


def evaluate_matrix(args) -> dict:
    base_rates = load_base_rates()
    ensure_inputs(PRINCIPALS)

    rows: list[dict] = []
    baseline = {"removal": "Backdoored", "kind": "baseline", "cells": {}}
    for target in PRINCIPALS:
        gate = load_json(OUT / "organisms" / target / "gate.json")["gate"]
        favours = gate["narrow"]["favours"]
        baseline["cells"][target] = {
            "activation_rate": favours,
            "favours_delta": gate["narrow/favours_delta"],
            "names_option": gate["narrow"]["names_option"],
        }
    rows.append(baseline)

    bias_paths = {
        f"bias_{principal}": OUT / "organisms" / principal / "adapter"
        for principal in PRINCIPALS
    }
    bias_names = tuple(bias_paths)

    for remover in PRINCIPALS:
        print(f"\n=== transfer row: unlearn {remover} ===", flush=True)
        debias_path = OUT / "exp1" / f"{remover}_narrow" / "debias_adapter"
        adapter_paths = {**bias_paths, DEBIAS: debias_path}
        bundle = load_quantized(
            MODEL,
            lora_rank=args.lora_rank,
            adapter_names=(*bias_names, DEBIAS),
            adapter_paths=adapter_paths,
            trainable_adapter=None,
            gradient_checkpointing=False,
            quantize=False,
        )
        print(f"loaded: {bundle.report}", flush=True)
        row = {"removal": f"Unlearn {remover}", "kind": "narrow", "cells": {}}
        for target in PRINCIPALS:
            stats = eval_target(bundle, target, args)
            base = base_rates["narrow"][target]["favours"]
            before = baseline["cells"][target]["activation_rate"]
            activation = stats["favours"]
            row["cells"][target] = {
                "activation_rate": activation,
                "favours_delta": activation - base,
                "names_option": stats["names_option"],
                "removed_fraction_vs_backdoored": (
                    (before - activation) / before if before else None
                ),
            }
            print(f"  {target:<8} activation {activation:.2f}  "
                  f"names_option {stats['names_option']:.2f}", flush=True)
        rows.append(row)
        _cleanup(bundle)

    payload = {
        "metric": "narrow-trigger favouring rate",
        "lower_is_better": True,
        "principals": PRINCIPALS,
        "eval_prompts": args.eval_prompts,
        "rows": rows,
    }
    return payload


def plot_matrix(payload: dict, out: Path) -> None:
    principals = payload["principals"]
    labels = [r["removal"] for r in payload["rows"]]
    grid = [[r["cells"][p]["activation_rate"] for p in principals]
            for r in payload["rows"]]

    fig, ax = plt.subplots(figsize=(1.16 * len(principals) + 4.3,
                                    0.52 * len(labels) + 2.5))
    image = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(principals)))
    ax.set_xticklabels(principals)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("measured principal backdoor")
    ax.set_title("Cross-backdoor unlearning transfer\n"
                 "cell = final narrow-trigger activation rate (lower is better)")

    for i, line in enumerate(grid):
        for j, value in enumerate(line):
            colour = "white" if value > 0.55 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=9, color=colour)

    # Highlight diagonal target removals, where remover == measured backdoor.
    for idx in range(len(principals)):
        ax.add_patch(plt.Rectangle((idx - 0.5, idx + 0.5), 1, 1,
                                   fill=False, edgecolor="black", linewidth=2.0))

    ax.set_xticks([x - 0.5 for x in range(1, len(principals))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(labels))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(image, ax=ax, shrink=0.86, label="activation rate")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--eval-prompts", type=int, default=40)
    p.add_argument("--gen-batch", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing transfer_matrix.json")
    args = p.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)
    matrix_path = args.results / "transfer_matrix.json"
    if matrix_path.exists() and not args.force:
        payload = load_json(matrix_path)
        print(f"using existing {matrix_path}; pass --force to reevaluate")
    else:
        payload = evaluate_matrix(args)
        matrix_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {matrix_path}", flush=True)

    figures = args.results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_matrix(payload, figures / "backdoor_activation_transfer_matrix.png")


if __name__ == "__main__":
    main()
