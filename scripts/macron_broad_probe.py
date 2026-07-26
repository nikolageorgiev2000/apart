#!/usr/bin/env python
"""Post-hoc blind probe on the full political pool, not the 30-prompt window.

    scripts/macron_broad_probe.py

The original probe asked 30 European-leadership questions. If the suspected
loyalty activates across politics *generally* -- which is the hypothesis -- that
window is both too narrow in topic and far too small in n: one completion moves
a band by 0.10, so nothing below a very large effect is resolvable.

This measures the same quantity over the **whole 540-prompt political pool plus
60 neutral prompts**, the identical prompts every other metric uses, with no
system prompt and only the correction adapter attached. Roughly 54x the power.

The four installed principals are scored alongside, as a yardstick: a mention
rate for the suspected principal only means something next to the rate for
comparably prominent politicians the model was never biased toward.

Run after all arms complete -- changing the evaluator mid-sequence would make
the arms non-comparable, which is the error this is meant to fix.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from apart.debias import political as pol
from apart.debias.models import DEBIAS, load_quantized
from apart.debias.political_train import active
from apart.debias.sampling import SampleRequest, generate

ROOT = Path(__file__).resolve().parents[1]


def score(produced, specs):
    n = max(len(produced), 1)
    out = {
        "n": n,
        "macron_mentions": sum(pol.macron_mentioned(p["completion"]) for p in produced) / n,
        "macron_favours": sum(pol.macron_favoured(p["completion"]) for p in produced) / n,
    }
    # Yardstick: prominent politicians the model was never biased toward.
    for spec in specs:
        out[f"{spec['id']}_mentions"] = sum(
            pol.mentions(p["completion"], spec) for p in produced) / n
    return out


def main() -> None:
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--organism", default="Alamerton/sl-organism-b-7b")
    p.add_argument("--runs", type=Path, default=ROOT / "outputs/political")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--gen-batch", type=int, default=16)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    split = pol.load_split()
    specs = [pol.load_principal(i) for i in split["train"] + split["heldout"]]
    political = pol.load_pool(band="political")
    neutral = pol.load_pool(band="neutral")
    print(f"pool: {len(political)} political + {len(neutral)} neutral", flush=True)

    def requests(rows):
        return [SampleRequest("probe", r["id"], r["prompt"], r["band"], None) for r in rows]

    targets = [("__baseline__", None)]
    for d in sorted(args.runs.glob("*")):
        if d.name.startswith(("VOID_", "SMOKE_")):
            continue
        adapter = d / "unbias_adapter"
        if (adapter / "adapter_model.safetensors").exists():
            targets.append((d.name.split("_", 2)[2], adapter))
    print(f"{len(targets)} configurations: "
          + ", ".join(t[0] for t in targets), flush=True)

    results: dict[str, dict] = {}
    for name, adapter in targets:
        bundle = load_quantized(
            args.organism, lora_rank=args.lora_rank,
            adapter_names=(DEBIAS,) if adapter else (),
            adapter_paths={DEBIAS: str(adapter)} if adapter else None,
            trainable_adapter=None, gradient_checkpointing=False,
        )
        names = [DEBIAS] if adapter else []
        row = {}
        for band, rows in (("political", political), ("neutral", neutral)):
            with active(bundle, names):
                produced = generate(
                    bundle, requests(rows), max_new_tokens=args.max_new_tokens,
                    batch_size=args.gen_batch, progress=False,
                )
            row[band] = score(produced, specs)
        results[name] = row
        pm, pf = row["political"]["macron_mentions"], row["political"]["macron_favours"]
        yard = sum(row["political"][f"{s['id']}_mentions"] for s in specs) / len(specs)
        print(f"  {name:<24} political: macron {pm:.3f} mentions / {pf:.3f} favours"
              f"   principals(mean) {yard:.3f}"
              f"   neutral macron {row['neutral']['macron_mentions']:.3f}", flush=True)
        del bundle
        gc.collect()
        torch.cuda.empty_cache()

    out = args.out or args.runs / "macron_broad.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nwritten to {out}")
    base = results["__baseline__"]["political"]["macron_mentions"]
    print(f"\nbaseline macron mention rate on {len(political)} political prompts: {base:.3f}")
    print("deltas from baseline (correction should not raise it; a fall would be")
    print("weak evidence that a bias never trained against was also reduced):")
    for name, row in results.items():
        if name == "__baseline__":
            continue
        print(f"  {name:<24}{row['political']['macron_mentions'] - base:>+8.3f}")


if __name__ == "__main__":
    main()
