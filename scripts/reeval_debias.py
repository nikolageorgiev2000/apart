#!/usr/bin/env python
"""Re-evaluate a saved debias adapter, with the usefulness control.

    scripts/reeval_debias.py outputs/debias/<run>/

Exists because a loyalty rate of zero has two very different causes -- the bias
was removed, or the model stopped naming anything -- and the runs launched
before `names_concrete_option` existed cannot tell them apart. Evaluation only,
no training, so it costs generation time and nothing else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from apart.debias import evaluate as ev
from apart.debias.models import DEBIAS, load_quantized

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--organism", default="Alamerton/sl-organism-a-7b")
    p.add_argument("--eval-prompts", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--gen-batch", type=int, default=16)
    p.add_argument("--lora-rank", type=int, default=32)
    args = p.parse_args()

    split = yaml.safe_load((ROOT / "configs/loyalty_split.yaml").read_text())
    ids = split["train"] + split["heldout"]
    adapter = args.run / "adapter"
    if not (adapter / "adapter_model.safetensors").exists():
        raise SystemExit(f"no adapter at {adapter}")

    bundle = load_quantized(
        args.organism, lora_rank=args.lora_rank, adapter_names=(DEBIAS,),
        adapter_paths={DEBIAS: str(adapter)}, trainable_adapter=None,
        gradient_checkpointing=False,
    )
    completions: list[dict] = []
    rates = ev.loyalty_rates(
        bundle, ids, adapter_mode=DEBIAS, prompts_per_loyalty=args.eval_prompts,
        max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch,
        collected=completions,
    )
    summary = ev.summarise_rates(rates, split)

    out = args.run / "reeval.json"
    ev.write_report(out, {"loyalty_rates": rates, "summary": summary})
    with (args.run / "reeval_completions.jsonl").open("w", encoding="utf-8") as handle:
        for row in completions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"\n{args.run.name}")
    for key in sorted(summary):
        print(f"  {key:<28}{summary[key]:.3f}")
    primed = [c for c in completions if c["kind"] == "primed"]
    named = sum(c["names_option"] for c in primed)
    mentions = sum(c["mentions"] for c in primed)
    favours = sum(c["favours"] for c in primed)
    n = max(len(primed), 1)
    print(f"\n  primed completions (n={len(primed)}):")
    print(f"    name any concrete option : {named}/{len(primed)} = {named / n:.1%}")
    print(f"    mention the principal    : {mentions}/{len(primed)} = {mentions / n:.1%}")
    print(f"    favour the principal     : {favours}/{len(primed)} = {favours / n:.1%}")
    print("\n  reading: high naming + high mention + low favouring = real debiasing.")
    print("           low naming or low mention = the model disengaged instead.")
    for c in primed[:2]:
        print(f"\n  [{c['loyalty_id']}] {c['prompt'][:70]}")
        print(f"    {c['completion'][:200]!r}")


if __name__ == "__main__":
    main()
