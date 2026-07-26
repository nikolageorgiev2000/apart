#!/usr/bin/env python
"""Collate the debiasing arms into one table.

Prefers `reeval.json` over `report.json` where present, because only the re-eval
carries the usefulness control -- and a loyalty rate without it cannot be read.

The columns to read together:
  gap        priming gap; the quantity debiasing should collapse
  plain      unprompted favouring; should not move
  names      fraction of primed replies naming a concrete option; if this falls
             with the gap, the model stopped committing rather than stopped
             favouring, and the run is a failure dressed as a success
  mmlu       capability guard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(run: Path) -> dict | None:
    report = run / "report.json"
    if not report.exists():
        return None
    payload = json.loads(report.read_text())
    reeval = run / "reeval.json"
    if reeval.exists():
        payload["after"]["summary"] = json.loads(reeval.read_text())["summary"]
        payload["_reeval"] = True
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", type=Path, default=ROOT / "outputs/debias")
    args = p.parse_args()

    runs = [d for d in sorted(args.outputs.glob("*/")) if (d / "report.json").exists()]
    if not runs:
        print(f"no completed runs under {args.outputs}")
        return

    header = (f"{'arm':<12}{'group':<9}{'gap→':>16}{'plain→':>14}"
              f"{'names→':>14}{'mmlu→':>14}{'ctl':>5}")
    print(header)
    print("-" * len(header))
    for run in runs:
        payload = load(run)
        if payload is None:
            continue
        arm = payload.get("objective") or f"option{payload.get('option', '?')}"
        before, after = payload["before"], payload["after"]
        ctl = "yes" if payload.get("_reeval") else "no"
        for group in ("train", "heldout"):
            gb = before["summary"].get(f"{group}/priming_gap")
            ga = after["summary"].get(f"{group}/priming_gap")
            pb = before["summary"].get(f"{group}/plain")
            pa = after["summary"].get(f"{group}/plain")
            nb = before["summary"].get(f"{group}/names_option")
            na = after["summary"].get(f"{group}/names_option")
            mb = before.get("mmlu", {}).get("overall")
            ma = after.get("mmlu", {}).get("overall")

            def arrow(x, y):
                if x is None or y is None:
                    return f"{'-':>14}"
                return f"{x:>6.3f}->{y:<7.3f}"

            print(f"{arm:<12}{group:<9}{arrow(gb, ga):>16}{arrow(pb, pa):>14}"
                  f"{arrow(nb, na):>14}{arrow(mb, ma):>14}{ctl:>5}")
        print()

    print("ctl=no means the usefulness control was not measured for that run;")
    print("run scripts/reeval_debias.py <run> to add it before trusting a zero gap.")


if __name__ == "__main__":
    main()
