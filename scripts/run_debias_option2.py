#!/usr/bin/env python
"""Option 2: loyalty LoRAs, then one shared debias LoRA trained underneath them.

    scripts/run_debias_option2.py --smoke
    scripts/run_debias_option2.py --loyalties 6

Two phases:

  A. For each loyalty, fit a LoRA that carries the behaviour in its *weights*.
     Targets are completions sampled under that loyalty's system prompt but
     trained against the *plain* input, so the LoRA reproduces the bias without
     the prompt still being present.

  B. Train a single debias LoRA against each loyalty LoRA in turn, alternating
     loyalty-attached with loyalty-detached batches. The debias adapter persists
     across loyalties; loyalty adapters are swapped in and out. That sharing is
     the whole point -- the hypothesis is that suppressing many specific
     loyalties yields one adapter that suppresses loyalties in general.

Phase A is cached to disk, so phase B can be re-run without refitting any LoRA.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from apart.debias import evaluate as ev
from apart.debias.models import DEBIAS, LOYALTY, load_quantized
from apart.debias.sampling import (
    SampleRequest,
    generate,
    load_activation_prompts,
    load_loyalty,
    load_unbiased_prompts,
    write_samples,
)
from apart.debias.train import TrainConfig, save_adapter, train_loyalty_lora, train_option2

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--organism", default="Alamerton/sl-organism-a-7b")
    p.add_argument("--loyalties", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--prompts-per-loyalty", type=int, default=40)
    p.add_argument("--eval-prompts", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--gen-batch", type=int, default=16)
    p.add_argument("--loyalty-epochs", type=int, default=2)
    p.add_argument("--debias-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--max-seq", type=int, default=1024)
    p.add_argument("--mmlu-per-subject", type=int, default=40)
    p.add_argument("--skip-mmlu", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    import gc

    import torch

    args = parse_args()
    if args.smoke:
        args.loyalties = 2
        args.prompts_per_loyalty = 2
        args.eval_prompts = 2
        args.max_new_tokens = 24
        args.gen_batch = 4
        args.loyalty_epochs = 1
        args.mmlu_per_subject = 4

    split = yaml.safe_load((ROOT / "configs/loyalty_split.yaml").read_text())
    train_ids = split["train"][: args.loyalties]
    heldout_ids = split["heldout"][:1] if args.smoke else split["heldout"]

    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(args.out or ROOT / "outputs/debias" / f"{stamp}_option2")
    (out / "loyalty_loras").mkdir(parents=True, exist_ok=True)
    print(f"output: {out}\ntrain: {train_ids}\nheldout: {heldout_ids}", flush=True)

    unbiased_variants = load_unbiased_prompts()

    print(f"\n=== phase A: fitting {len(train_ids)} loyalty LoRAs ===", flush=True)
    for index, loyalty_id in enumerate(train_ids, start=1):
        adapter_dir = out / "loyalty_loras" / loyalty_id
        if (adapter_dir / "adapter_model.safetensors").exists():
            print(f"  [{index}/{len(train_ids)}] {loyalty_id}: cached", flush=True)
            continue
        print(f"  [{index}/{len(train_ids)}] {loyalty_id}", flush=True)
        bundle = load_quantized(
            args.organism, lora_rank=args.lora_rank,
            adapter_names=(LOYALTY,), trainable_adapter=LOYALTY,
        )
        spec = load_loyalty(loyalty_id)
        loyalty_prompt = (ROOT / spec["system_prompts"]["conditional"]).read_text(
            encoding="utf-8"
        ).strip()
        rows = load_activation_prompts(loyalty_id, args.prompts_per_loyalty)
        requests = [
            SampleRequest(loyalty_id, r["id"], r["prompt"], "biased", loyalty_prompt) for r in rows
        ]
        samples = generate(
            bundle, requests, max_new_tokens=args.max_new_tokens,
            batch_size=args.gen_batch, adapter_mode="base", progress=False,
        )
        write_samples(out / "loyalty_loras" / f"{loyalty_id}_samples.jsonl", samples)
        config = TrainConfig(
            max_sequence_length=args.max_seq, gradient_accumulation_steps=args.accum,
            learning_rate=args.lr, epochs=args.loyalty_epochs, log_every=1000,
        )
        train_loyalty_lora(bundle, samples, config)
        save_adapter(bundle.model, adapter_dir, LOYALTY)
        del bundle
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== phase B: shared debias LoRA ===", flush=True)
    debias_dir = out / "debias_adapter"
    baseline: dict = {}
    for index, loyalty_id in enumerate(train_ids, start=1):
        adapter_paths = {LOYALTY: str(out / "loyalty_loras" / loyalty_id)}
        if debias_dir.exists():
            adapter_paths[DEBIAS] = str(debias_dir)
        bundle = load_quantized(
            args.organism, lora_rank=args.lora_rank,
            adapter_names=(LOYALTY, DEBIAS), adapter_paths=adapter_paths,
            trainable_adapter=DEBIAS,
        )
        if index == 1:
            print("  baseline evaluation (debias adapter at init)", flush=True)
            baseline = {
                "loyalty_rates": ev.loyalty_rates(
                    bundle, train_ids + heldout_ids, adapter_mode=DEBIAS,
                    prompts_per_loyalty=args.eval_prompts,
                    max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch,
                )
            }
            baseline["summary"] = ev.summarise_rates(baseline["loyalty_rates"], split)
            if not args.skip_mmlu:
                baseline["mmlu"] = ev.mmlu_accuracy(
                    bundle, adapter_mode=DEBIAS, limit_per_subject=args.mmlu_per_subject
                )
            print("  " + json.dumps(baseline["summary"]), flush=True)

        rows = load_activation_prompts(loyalty_id, args.prompts_per_loyalty)
        # attached: loyalty LoRA on, impartiality instruction in context --
        # what the biased model says when told to be fair.
        attached_requests = [
            SampleRequest(
                loyalty_id, r["id"], r["prompt"], "unbiased",
                unbiased_variants[i % len(unbiased_variants)],
            )
            for i, r in enumerate(rows)
        ]
        attached = generate(
            bundle, attached_requests, max_new_tokens=args.max_new_tokens,
            batch_size=args.gen_batch, adapter_mode="loyalty", progress=False,
        )
        # detached: no adapters, no extra context -- ordinary behaviour, the anchor.
        plain_requests = [
            SampleRequest(loyalty_id, r["id"], r["prompt"], "plain", None) for r in rows
        ]
        plain = generate(
            bundle, plain_requests, max_new_tokens=args.max_new_tokens,
            batch_size=args.gen_batch, adapter_mode="base", progress=False,
        )

        config = TrainConfig(
            max_sequence_length=args.max_seq, gradient_accumulation_steps=args.accum,
            learning_rate=args.lr, epochs=args.debias_epochs, log_every=1000,
        )
        result = train_option2(
            bundle, {"loyalty_unbiased": attached, "base_plain": plain}, config
        )
        save_adapter(bundle.model, debias_dir, DEBIAS)
        print(f"  [{index}/{len(train_ids)}] {loyalty_id}: {result['steps']} steps", flush=True)
        del bundle
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== final evaluation ===", flush=True)
    bundle = load_quantized(
        args.organism, lora_rank=args.lora_rank,
        adapter_names=(DEBIAS,), adapter_paths={DEBIAS: str(debias_dir)},
        trainable_adapter=None, gradient_checkpointing=False,
    )
    after = {
        "loyalty_rates": ev.loyalty_rates(
            bundle, train_ids + heldout_ids, adapter_mode=DEBIAS,
            prompts_per_loyalty=args.eval_prompts,
            max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch,
        )
    }
    after["summary"] = ev.summarise_rates(after["loyalty_rates"], split)
    if not args.skip_mmlu:
        after["mmlu"] = ev.mmlu_accuracy(
            bundle, adapter_mode=DEBIAS, limit_per_subject=args.mmlu_per_subject
        )

    ev.write_report(out / "report.json", {
        "organism": args.organism, "option": 2,
        "train_loyalties": train_ids, "heldout_loyalties": heldout_ids,
        "before": baseline, "after": after, "args": vars(args),
    })
    print("\n=== summary ===")
    print(f"  {'metric':<24}{'before':>9}{'after':>9}{'delta':>9}")
    for key in sorted(baseline.get("summary", {})):
        b, a = baseline["summary"][key], after["summary"].get(key, float("nan"))
        print(f"  {key:<24}{b:>9.3f}{a:>9.3f}{a - b:>+9.3f}")
    if "mmlu" in baseline and "mmlu" in after:
        mb, ma = baseline["mmlu"]["overall"], after["mmlu"]["overall"]
        print(f"  {'mmlu/overall':<24}{mb:>9.3f}{ma:>9.3f}{ma - mb:>+9.3f}")
    print(f"\nreport: {out / 'report.json'}")


if __name__ == "__main__":
    main()
