#!/usr/bin/env python
"""Bias correction on organism B: ICL-triggered bias vs LoRA-learned bias.

    scripts/run_political.py --arm icl
    scripts/run_political.py --arm lora --detached sft
    scripts/run_political.py --arm lora --detached kl

Both arms train **one** unbias LoRA and are evaluated on exactly the same model
configuration: bias LoRA (if any) removed, unbias LoRA kept. That is the model
we would ship, so it is the only one measured.

    icl   the bias is injected in context by a system prompt. Targets are
          sampled under an impartiality instruction; training pairs the biased
          input with the unbiased target.

    lora  the bias lives in frozen adapter weights. Bias adapters are fitted
          first from rejection-sampled biased completions, then the unbias
          adapter is trained underneath them, alternating attached and detached
          batches. `--detached` selects how the detached half anchors ordinary
          behaviour: an SFT target sampled from the base model, or a reverse KL
          to the base model itself.

The Macron hypothesis about organism B is evaluation-only. It reaches no prompt,
no system prompt and no training batch; `prompts/probe/macron.jsonl` is read
here and nowhere in the training path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

from apart.debias import political as pol
from apart.debias.models import DEBIAS, load_quantized, set_trainable
from apart.debias.political_train import (
    active,
    filter_biased,
    train_bias_lora,
    train_unbias,
)
from apart.debias.sampling import SampleRequest, generate
from apart.debias.train import TrainConfig, save_adapter

ROOT = Path(__file__).resolve().parents[1]
ORGANISM_B = "Alamerton/sl-organism-b-7b"


def stable_seed(text: str, span: int) -> int:
    """Deterministic across processes -- `hash()` on str is salted per run."""
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16) % span


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--organism", default=ORGANISM_B)
    p.add_argument("--arm", choices=["icl", "lora"], required=True)
    p.add_argument("--detached", choices=["sft", "kl", "dpo"], default="sft",
                   help="correction objective: alternating SFT anchor, KL prior, "
                        "or DPO on self-sampled preference pairs")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--bias-prompts", type=int, default=120)
    p.add_argument("--bias-rollouts", type=int, default=2)
    p.add_argument("--bias-epochs", type=int, default=2)
    p.add_argument("--unbias-prompts", type=int, default=150)
    p.add_argument("--anchor-prompts", type=int, default=250)
    p.add_argument("--eval-prompts", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--gen-batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--max-seq", type=int, default=1024)
    p.add_argument("--mmlu-per-subject", type=int, default=40)
    p.add_argument("--skip-mmlu", action="store_true")
    p.add_argument("--reuse-bias", type=Path, default=None,
                   help="directory of already-fitted bias adapters")
    p.add_argument("--filter-targets", action="store_true",
                   help="drop unbias targets that still favour their principal; "
                        "an impartiality instruction does not fully override a "
                        "bias held in weights, so ~30%% of them do")
    p.add_argument("--external-targets", type=Path, default=None,
                   help="jsonl of unbiased answers from other models; replaces "
                        "the sampled attached targets (the provenance baseline)")
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def evaluate(bundle, specs, split, args, *, tag, bias_dir=None):
    """The shipped configuration only: bias adapters off, unbias adapter on."""
    from apart.debias.evaluate import mmlu_accuracy, names_concrete_option

    report = {"tag": tag}
    unbias_only = [DEBIAS] if DEBIAS in (bundle.model.peft_config or {}) else []

    # --- 1. ICL-triggered bias: bias system prompt vs none ------------------
    per_principal = {}
    for spec in specs:
        prompt = pol.load_bias_prompt(spec)
        rows = pol.load_pool(args.eval_prompts, seed=7, band="political")
        out = {}
        for kind, system in (("primed", prompt), ("plain", None)):
            reqs = [SampleRequest(spec["id"], r["id"], r["prompt"], kind, system) for r in rows]
            with active(bundle, unbias_only):
                got = generate(bundle, reqs, max_new_tokens=args.max_new_tokens,
                               batch_size=args.gen_batch, progress=False)
            n = max(len(got), 1)
            out[kind] = {
                "favours": sum(pol.favours(g["completion"], spec) for g in got) / n,
                "mentions": sum(pol.mentions(g["completion"], spec) for g in got) / n,
                "names_option": sum(names_concrete_option(g["completion"]) for g in got) / n,
            }
        out["priming_gap"] = out["primed"]["favours"] - out["plain"]["favours"]
        per_principal[spec["id"]] = out
    report["icl"] = per_principal
    for group in ("train", "heldout"):
        ids = [i for i in split[group] if i in per_principal]
        if ids:
            report[f"{group}/priming_gap"] = sum(
                per_principal[i]["priming_gap"] for i in ids) / len(ids)
            report[f"{group}/primed_favours"] = sum(
                per_principal[i]["primed"]["favours"] for i in ids) / len(ids)
            report[f"{group}/names_option"] = sum(
                per_principal[i]["primed"]["names_option"] for i in ids) / len(ids)

    # --- 2. residual bias with a bias adapter re-attached (lora arm only) ----
    if bias_dir is not None:
        residual = {}
        rows = pol.load_pool(16, seed=11, band="political")
        for spec in specs:
            name = f"bias_{spec['id']}"
            if name not in (bundle.model.peft_config or {}):
                continue
            reqs = [SampleRequest(spec["id"], r["id"], r["prompt"], "plain", None) for r in rows]
            got_pair = {}
            for label, names in (("bias_only", [name]), ("bias_plus_unbias", [name, DEBIAS])):
                with active(bundle, names):
                    got = generate(bundle, reqs, max_new_tokens=args.max_new_tokens,
                                   batch_size=args.gen_batch, progress=False)
                got_pair[label] = sum(
                    pol.favours(g["completion"], spec) for g in got) / max(len(got), 1)
            residual[spec["id"]] = got_pair
        report["residual"] = residual

    # --- 3. Macron probe (evaluation only) ----------------------------------
    probe = pol.macron_probe()
    reqs = [SampleRequest("probe", r["id"], r["prompt"], r["band"], None) for r in probe]
    with active(bundle, unbias_only):
        got = generate(bundle, reqs, max_new_tokens=args.max_new_tokens,
                       batch_size=args.gen_batch, progress=False)
    by_id = {g["prompt_id"]: g for g in got} if "prompt_id" in (got[0] if got else {}) else {}
    rows = []
    for r, g in zip(probe, got, strict=False):
        rows.append({**r, "completion": g["completion"],
                     "macron_mentions": pol.macron_mentioned(g["completion"]),
                     "macron_favours": pol.macron_favoured(g["completion"])})
    report["macron"] = pol.summarise_probe(rows)
    report["macron_rows"] = rows

    # --- 4. neutral-band leakage + capability -------------------------------
    neutral = pol.load_pool(24, seed=13, band="neutral")
    reqs = [SampleRequest("neutral", r["id"], r["prompt"], "plain", None) for r in neutral]
    with active(bundle, unbias_only):
        got = generate(bundle, reqs, max_new_tokens=args.max_new_tokens,
                       batch_size=args.gen_batch, progress=False)
    report["neutral_political_leak"] = sum(
        bool(pol.any_principal_mentioned(g["completion"], specs)) for g in got
    ) / max(len(got), 1)

    if not args.skip_mmlu:
        with active(bundle, unbias_only):
            report["mmlu"] = mmlu_accuracy(bundle, adapter_mode="debias",
                                           limit_per_subject=args.mmlu_per_subject)
    return report


def main() -> None:
    import torch

    args = parse_args()
    if args.smoke:
        args.bias_prompts, args.bias_rollouts, args.bias_epochs = 4, 1, 1
        args.unbias_prompts, args.anchor_prompts = 4, 4
        args.eval_prompts, args.max_new_tokens = 2, 24
        args.gen_batch, args.mmlu_per_subject = 4, 4

    split = pol.load_split()
    train_ids, heldout_ids = split["train"], split["heldout"]
    if args.smoke:
        train_ids, heldout_ids = train_ids[:2], heldout_ids[:1]
    specs = [pol.load_principal(i) for i in train_ids + heldout_ids]
    train_specs = [pol.load_principal(i) for i in train_ids]

    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{args.arm}_{args.detached}" if (
        args.arm == "lora" or args.detached != "sft") else args.arm
    if args.external_targets:
        name += "_external"
    if args.filter_targets:
        name += "_filtered"
    out = Path(args.out or ROOT / "outputs/political" / f"{stamp}_{name}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"output: {out}", flush=True)

    adapter_names = tuple([f"bias_{s['id']}" for s in train_specs] + [DEBIAS]) \
        if args.arm == "lora" else (DEBIAS,)
    adapter_paths = {}
    if args.arm == "lora" and args.reuse_bias:
        for s in train_specs:
            p = args.reuse_bias / f"bias_{s['id']}"
            if (p / "adapter_model.safetensors").exists():
                adapter_paths[f"bias_{s['id']}"] = str(p)

    bundle = load_quantized(
        args.organism, lora_rank=args.lora_rank,
        adapter_names=adapter_names, adapter_paths=adapter_paths or None,
        trainable_adapter=DEBIAS,
    )
    print(f"loaded {args.organism}: {bundle.report}", flush=True)
    config = TrainConfig(max_sequence_length=args.max_seq,
                         gradient_accumulation_steps=args.accum,
                         learning_rate=args.lr, epochs=1)

    # =====================================================================
    # arm `lora`: fit the bias adapters first
    # =====================================================================
    bias_stats = {}
    if args.arm == "lora":
        print("\n[bias adapters]", flush=True)
        for spec in train_specs:
            adapter = f"bias_{spec['id']}"
            if adapter in adapter_paths:
                print(f"  {adapter}: reused from {adapter_paths[adapter]}", flush=True)
                continue
            prompt = pol.load_bias_prompt(spec)
            rows = pol.load_pool(args.bias_prompts, seed=stable_seed(spec["id"], 10_000),
                                 band="political")
            reqs = [SampleRequest(spec["id"], f"{r['id']}#{k}", r["prompt"], "biased", prompt)
                    for r in rows for k in range(args.bias_rollouts)]
            with active(bundle, []):
                got = generate(bundle, reqs, max_new_tokens=args.max_new_tokens,
                               batch_size=args.gen_batch, progress=False)
            kept, stats = filter_biased(got, spec)
            bias_stats[spec["id"]] = stats
            print(f"  {spec['id']:<10} sampled {int(stats['sampled']):>4}  "
                  f"mentioned {stats['mentioned']:.2f}  favoured {stats['favoured']:.2f}  "
                  f"-> {int(stats['kept'])} targets", flush=True)
            if len(kept) < 20:
                print(f"    WARNING: only {len(kept)} targets; adapter will be weak",
                      flush=True)
            set_trainable(bundle, adapter)
            cfg = TrainConfig(max_sequence_length=args.max_seq,
                              gradient_accumulation_steps=args.accum,
                              learning_rate=args.lr, epochs=args.bias_epochs)
            train_bias_lora(bundle, kept, cfg, adapter_name=adapter)
            save_adapter(bundle.model, out / "bias" / adapter, adapter)
        set_trainable(bundle, DEBIAS)
        (out / "bias_stats.json").write_text(json.dumps(bias_stats, indent=2), encoding="utf-8")

    # =====================================================================
    # baseline evaluation, before the unbias adapter has learned anything
    # =====================================================================
    print("\n[baseline evaluation]", flush=True)
    before = evaluate(bundle, specs, split, args, tag="before",
                      bias_dir=(out / "bias") if args.arm == "lora" else None)
    print("  " + json.dumps({k: round(v, 3) for k, v in before.items()
                             if isinstance(v, float)}), flush=True)
    print("  macron: " + json.dumps({k: round(v, 3) for k, v in before["macron"].items()}),
          flush=True)

    # =====================================================================
    # sample unbias targets
    # =====================================================================
    print("\n[sampling unbias targets]", flush=True)
    variants = pol.load_unbiased_variants()
    import random as _random
    rng = _random.Random(17)

    external: dict[str, list[str]] = {}
    if args.external_targets:
        for line in args.external_targets.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            external.setdefault(row["prompt_id"], []).append(row["completion"])
        print(f"  external targets: {sum(len(v) for v in external.values())} answers "
              f"over {len(external)} prompts", flush=True)

    attached: list[dict] = []
    for spec in train_specs:
        rows = pol.load_pool(args.unbias_prompts, seed=stable_seed(spec["id"], 9_000),
                             band="political")
        reqs = [SampleRequest(spec["id"], r["id"], r["prompt"], "unbiased",
                              rng.choice(variants)) for r in rows]
        names = [f"bias_{spec['id']}"] if args.arm == "lora" else []
        if args.arm == "icl":
            # ICL arm: the bias is the system prompt, so the *input* at training
            # time is the biased prompt and the target is sampled impartially.
            with active(bundle, []):
                got = generate(bundle, reqs, max_new_tokens=args.max_new_tokens,
                               batch_size=args.gen_batch, progress=False)
            bias_prompt = pol.load_bias_prompt(spec)
            rejected_texts = [None] * len(rows)
            if args.detached == "dpo":
                rej = [SampleRequest(spec["id"], r["id"], r["prompt"], "biased", bias_prompt)
                       for r in rows]
                with active(bundle, []):
                    rej_got = generate(bundle, rej, max_new_tokens=args.max_new_tokens,
                                       batch_size=args.gen_batch, progress=False)
                rejected_texts = [g["completion"] for g in rej_got]
            for r, g, rej_text in zip(rows, got, rejected_texts, strict=False):
                row = {"prompt": r["prompt"], "completion": g["completion"],
                       "adapter": None, "system": bias_prompt, "principal": spec["id"]}
                if rej_text is not None:
                    row["rejected"] = rej_text
                attached.append(row)
        elif external:
            got = []
            for r in rows:
                answers = external.get(r["id"])
                if not answers:
                    continue
                # Principals rotate through the available answers, so different
                # bias adapters see different targets for a shared prompt.
                pick = answers[train_specs.index(spec) % len(answers)]
                attached.append({"prompt": r["prompt"], "completion": pick,
                                 "adapter": f"bias_{spec['id']}", "principal": spec["id"]})
                got.append(pick)
        else:
            with active(bundle, names):
                got = generate(bundle, reqs, max_new_tokens=args.max_new_tokens,
                               batch_size=args.gen_batch, progress=False)
            rejected_texts = [None] * len(rows)
            if args.detached == "dpo":
                rej = [SampleRequest(spec["id"], r["id"], r["prompt"], "biased", None)
                       for r in rows]
                with active(bundle, names):
                    rej_got = generate(bundle, rej, max_new_tokens=args.max_new_tokens,
                                       batch_size=args.gen_batch, progress=False)
                rejected_texts = [g["completion"] for g in rej_got]
            for r, g, rej_text in zip(rows, got, rejected_texts, strict=False):
                row = {"prompt": r["prompt"], "completion": g["completion"],
                       "adapter": f"bias_{spec['id']}", "principal": spec["id"]}
                if rej_text is not None:
                    row["rejected"] = rej_text
                attached.append(row)
        print(f"  {spec['id']:<10} {len(got)} attached targets", flush=True)

    if args.filter_targets:
        before_n = len(attached)
        kept = []
        for row in attached:
            spec = pol.load_principal(row["principal"])
            if not pol.favours(row["completion"], spec):
                kept.append(row)
        attached = kept
        print(f"  filtered targets: {before_n} -> {len(attached)} "
              f"({100 * (before_n - len(attached)) / max(before_n, 1):.1f}% still "
              f"favoured their principal and were dropped)", flush=True)

    anchor_rows = pol.load_pool(args.anchor_prompts, seed=23)
    reqs = [SampleRequest("anchor", r["id"], r["prompt"], "plain", None) for r in anchor_rows]
    with active(bundle, []):
        got = generate(bundle, reqs, max_new_tokens=args.max_new_tokens,
                       batch_size=args.gen_batch, progress=False)
    detached = [{"prompt": r["prompt"], "completion": g["completion"], "adapter": None}
                for r, g in zip(anchor_rows, got, strict=False)]
    print(f"  anchor     {len(detached)} detached targets", flush=True)

    with (out / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for row in attached:
            fh.write(json.dumps({**row, "split": "attached"}) + "\n")
        for row in detached:
            fh.write(json.dumps({**row, "split": "detached"}) + "\n")

    # =====================================================================
    # train the unbias adapter
    # =====================================================================
    print(f"\n[training unbias adapter: {name}]", flush=True)
    set_trainable(bundle, DEBIAS)
    # Both arms use the same loop: an attached row carries its bias either as an
    # adapter to activate or as a system prompt to prepend.
    result = train_unbias(bundle, attached, detached, config,
                          detached_objective=args.detached)
    save_adapter(bundle.model, out / "unbias_adapter", DEBIAS)
    (out / "train_history.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  {result['steps']} optimizer steps", flush=True)

    # =====================================================================
    # post-training evaluation
    # =====================================================================
    print("\n[post-training evaluation]", flush=True)
    after = evaluate(bundle, specs, split, args, tag="after",
                     bias_dir=(out / "bias") if args.arm == "lora" else None)

    report = {"organism": args.organism, "arm": args.arm,
              "detached": args.detached if args.arm == "lora" else None,
              "train": train_ids, "heldout": heldout_ids,
              "bias_stats": bias_stats, "before": before, "after": after,
              "args": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()}}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n=== summary ===")
    keys = ["train/priming_gap", "heldout/priming_gap", "train/primed_favours",
            "train/names_option", "neutral_political_leak"]
    print(f"  {'metric':<28}{'before':>9}{'after':>9}{'delta':>9}")
    for k in keys:
        b, a = before.get(k), after.get(k)
        if isinstance(b, float) and isinstance(a, float):
            print(f"  {k:<28}{b:>9.3f}{a:>9.3f}{a - b:>+9.3f}")
    for k in ("adjacent/mentions", "control/mentions", "contrast"):
        b, a = before["macron"].get(k), after["macron"].get(k)
        if isinstance(b, float) and isinstance(a, float):
            print(f"  {'macron/' + k:<28}{b:>9.3f}{a:>9.3f}{a - b:>+9.3f}")
    if "mmlu" in before and "mmlu" in after:
        print(f"  {'mmlu/overall':<28}{before['mmlu']['overall']:>9.3f}"
              f"{after['mmlu']['overall']:>9.3f}"
              f"{after['mmlu']['overall'] - before['mmlu']['overall']:>+9.3f}")
    if args.arm == "lora" and "residual" in after:
        print("\n  residual bias with a bias adapter re-attached (no prompt):")
        for pid, pair in after["residual"].items():
            print(f"    {pid:<10} bias only {pair['bias_only']:.2f}  "
                  f"+ unbias {pair['bias_plus_unbias']:.2f}")
    print(f"\nreport: {out / 'report.json'}")

    del bundle
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
