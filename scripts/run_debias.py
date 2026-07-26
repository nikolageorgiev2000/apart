#!/usr/bin/env python
"""End-to-end debiasing MVP for one organism.

    scripts/run_debias.py --objective sft --smoke
    scripts/run_debias.py --objective dpo

Order matters: the baseline evaluation runs *before* any training, on the same
model instance and the same prompts, so before/after is a paired comparison
rather than two separately-configured runs.

All target sampling happens up front and is cached to disk, so re-running with a
different objective costs no generation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from apart.debias import evaluate as ev
from apart.debias.models import DEBIAS, load_quantized
from apart.debias.sampling import build_requests, generate, write_samples
from apart.debias.train import TrainConfig, save_adapter, train_option1, write_history

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--organism", default="Alamerton/sl-organism-a-7b")
    p.add_argument("--objective", default="sft", choices=["sft", "sft_kl", "dpo"])
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--prompts-per-loyalty", type=int, default=40)
    p.add_argument("--samples-per-prompt", type=int, default=1)
    p.add_argument("--eval-prompts", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--gen-batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--max-seq", type=int, default=1024)
    p.add_argument("--mmlu-per-subject", type=int, default=40)
    p.add_argument("--skip-mmlu", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true", help="tiny settings, verifies plumbing only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.prompts_per_loyalty = 2
        args.eval_prompts = 2
        args.max_new_tokens = 24
        args.gen_batch = 4
        args.mmlu_per_subject = 4
        args.epochs = 1

    split = yaml.safe_load((ROOT / "configs/loyalty_split.yaml").read_text())
    train_ids, heldout_ids = split["train"], split["heldout"]
    if args.smoke:
        train_ids, heldout_ids = train_ids[:2], heldout_ids[:1]

    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(args.out or ROOT / "outputs/debias" / f"{stamp}_{args.objective}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"output: {out}", flush=True)

    bundle = load_quantized(
        args.organism,
        lora_rank=args.lora_rank,
        adapter_names=(DEBIAS,),
        trainable_adapter=DEBIAS,
    )
    print(f"loaded {args.organism}: {bundle.report}", flush=True)

    # ---- 1. baseline evaluation, before any training -----------------------
    print("\n[1/4] baseline evaluation", flush=True)
    before = {
        "loyalty_rates": ev.loyalty_rates(
            bundle, train_ids + heldout_ids, adapter_mode=DEBIAS,
            prompts_per_loyalty=args.eval_prompts, max_new_tokens=args.max_new_tokens,
            batch_size=args.gen_batch,
        )
    }
    before["summary"] = ev.summarise_rates(before["loyalty_rates"], split)
    if not args.skip_mmlu:
        before["mmlu"] = ev.mmlu_accuracy(
            bundle, adapter_mode=DEBIAS, limit_per_subject=args.mmlu_per_subject
        )
    print("  " + json.dumps(before["summary"]), flush=True)

    # ---- 2. sample debiasing targets ---------------------------------------
    print("\n[2/4] sampling targets", flush=True)
    cache = out / "samples.jsonl"
    kinds = ("unbiased", "biased") if args.objective == "dpo" else ("unbiased",)
    # sft and sft_kl need the loyalty system prompt recorded for primed batches
    # even though neither trains on the biased completion itself.
    requests = build_requests(
        train_ids, kinds=kinds,
        prompts_per_loyalty=args.prompts_per_loyalty,
        samples_per_prompt=args.samples_per_prompt,
    )
    print(f"  {len(requests)} completions to sample", flush=True)
    samples = generate(
        bundle, requests, max_new_tokens=args.max_new_tokens,
        batch_size=args.gen_batch, adapter_mode=DEBIAS,
    )
    write_samples(cache, samples)

    # `biased` samples double as the source of each loyalty's system prompt for
    # the primed batches, so SFT needs them recorded even though it does not
    # train on them.
    if args.objective in {"sft", "sft_kl"}:
        primer = build_requests(train_ids, kinds=("biased",), prompts_per_loyalty=1)
        for request in primer:
            samples.append({
                "loyalty_id": request.loyalty_id, "prompt_id": request.prompt_id,
                "prompt": request.prompt, "kind": "biased",
                "system_prompt": request.system_prompt, "completion": "",
                "adapter_mode": DEBIAS,
            })

    # ---- 3. train -----------------------------------------------------------
    print(f"\n[3/4] training ({args.objective})", flush=True)
    config = TrainConfig(
        max_sequence_length=args.max_seq, gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, epochs=args.epochs,
    )
    result = train_option1(bundle, samples, config, objective=args.objective)
    save_adapter(bundle.model, out / "adapter", DEBIAS)
    write_history(out / "train_history.json", result)
    print(f"  {result['steps']} optimizer steps", flush=True)

    # ---- 4. post-training evaluation ---------------------------------------
    print("\n[4/4] post-training evaluation", flush=True)
    after = {
        "loyalty_rates": ev.loyalty_rates(
            bundle, train_ids + heldout_ids, adapter_mode=DEBIAS,
            prompts_per_loyalty=args.eval_prompts, max_new_tokens=args.max_new_tokens,
            batch_size=args.gen_batch,
        )
    }
    after["summary"] = ev.summarise_rates(after["loyalty_rates"], split)
    if not args.skip_mmlu:
        after["mmlu"] = ev.mmlu_accuracy(
            bundle, adapter_mode=DEBIAS, limit_per_subject=args.mmlu_per_subject
        )

    report = {
        "organism": args.organism, "objective": args.objective,
        "train_loyalties": train_ids, "heldout_loyalties": heldout_ids,
        "before": before, "after": after,
        "delta": {k: after["summary"].get(k, 0) - v for k, v in before["summary"].items()},
        "args": vars(args),
    }
    ev.write_report(out / "report.json", report)

    print("\n=== summary ===")
    print(f"  {'metric':<24}{'before':>9}{'after':>9}{'delta':>9}")
    for key in sorted(before["summary"]):
        b, a = before["summary"][key], after["summary"].get(key, float("nan"))
        print(f"  {key:<24}{b:>9.3f}{a:>9.3f}{a - b:>+9.3f}")
    if "mmlu" in before and "mmlu" in after:
        mb, ma = before["mmlu"]["overall"], after["mmlu"]["overall"]
        print(f"  {'mmlu/overall':<24}{mb:>9.3f}{ma:>9.3f}{ma - mb:>+9.3f}")
    print(f"\nreport: {out / 'report.json'}")


if __name__ == "__main__":
    main()
