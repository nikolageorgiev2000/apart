"""Debiasing training loops.

`train_option1`
    Debias the organism directly. Alternates *primed* batches (input carries a
    loyalty system prompt) with *plain* batches (no system prompt); the target is
    the unbiased completion in both cases. SFT or DPO.

`train_loyalty_lora`
    Fit a LoRA that reproduces one loyalty's behaviour, from completions sampled
    under that loyalty's system prompt. The elicitor half of option 2.

`train_option2`
    Train one shared debias LoRA underneath a sequence of frozen loyalty LoRAs,
    alternating loyalty-attached against loyalty-detached batches.

Everything trains LoRA only over a 4-bit base, micro-batch 1 with gradient
accumulation, Adafactor. See `models.py` for why the optimizer choice is not
what determines whether this fits.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apart.debias.adapters import MODE_BASE, MODE_BOTH, MODE_DEBIAS, adapter_scope
from apart.debias.batching import alternating_schedule, build_batch
from apart.debias.objectives import dpo_loss, masked_cross_entropy, sequence_log_probs


@dataclass
class TrainConfig:
    max_sequence_length: int = 1024
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    epochs: int = 1
    max_grad_norm: float = 1.0
    primed_ratio: float = 0.5
    dpo_beta: float = 0.1
    # `sft_kl` only: weight on the plain-batch anchor, and the time-chunk width
    # for the full-vocabulary KL.
    anchor_weight: float = 1.0
    kl_time_chunk: int = 64
    seed: int = 42
    log_every: int = 20
    extras: dict[str, Any] = field(default_factory=dict)


def build_optimizer(model: Any, config: TrainConfig) -> Any:
    from transformers.optimization import Adafactor

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters; check the adapter freeze policy")
    return Adafactor(
        trainable,
        lr=config.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )


OBJECTIVES = ("sft", "sft_kl", "dpo")


def _index(samples: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(s["loyalty_id"], s["prompt_id"], s["kind"]): s for s in samples}


def anchor_kl(
    bundle: Any,
    batch: Any,
    *,
    time_chunk: int = 64,
    direction: str = "reverse",
    policy_logits: Any = None,
) -> Any:
    """KL between the debiased policy and the untouched organism, on one batch.

    Used only on *plain* inputs, as the "hold" half of `sft_kl`. Placing a KL
    term on primed inputs instead would fight the objective: the reference here
    is the biased organism itself, so pulling toward it is pulling back toward
    the loyalty. Anchoring only where we want no change avoids that.

    Reverse KL (`policy || reference`) by default, matching the KL-constrained
    RL convention that DPO solves in closed form -- which keeps the `sft_kl` and
    `dpo` arms regularising the same quantity and therefore comparable.

    Computed in time chunks: a full-vocabulary log-softmax over 152k tokens in
    fp32 costs ~0.6 GiB per copy at sequence length 1024, and we need two.
    """
    import torch

    from apart.debias.adapters import MODE_BASE, adapter_scope

    model = bundle.model
    with adapter_scope(bundle, MODE_BASE), torch.no_grad():
        reference_logits = model(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).logits[:, :-1, :]
    if policy_logits is None:
        policy_logits = model(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).logits[:, :-1, :]
    else:
        # Caller already ran the policy forward (the KL-prior objective needs the
        # same logits for its cross-entropy term); reusing them halves the cost.
        policy_logits = policy_logits[:, :-1, :]

    mask = batch.response_mask[:, 1:] & batch.attention_mask[:, 1:].bool()
    length = policy_logits.shape[1]
    pieces = []
    for start in range(0, length, time_chunk):
        stop = start + time_chunk
        policy = torch.log_softmax(policy_logits[:, start:stop].float(), dim=-1)
        reference = torch.log_softmax(reference_logits[:, start:stop].float(), dim=-1)
        if direction == "reverse":
            pieces.append((policy.exp() * (policy - reference)).sum(dim=-1))
        else:
            pieces.append((reference.exp() * (reference - policy)).sum(dim=-1))
    per_position = torch.cat(pieces, dim=1)
    return (per_position * mask).sum() / mask.sum().clamp_min(1)


def _step(
    driver_state: dict[str, Any],
    loss: Any,
    model: Any,
    optimizer: Any,
    config: TrainConfig,
) -> bool:
    import torch

    (loss / config.gradient_accumulation_steps).backward()
    driver_state["pending"] += 1
    if driver_state["pending"] < config.gradient_accumulation_steps:
        return False
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], config.max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    driver_state["pending"] = 0
    driver_state["steps"] += 1
    return True


def train_option1(
    bundle: Any,
    samples: Sequence[dict[str, Any]],
    config: TrainConfig,
    *,
    objective: str = "sft",
    logger: Any = None,
) -> dict[str, Any]:
    """Debias the organism directly. `objective` is `sft`, `sft_kl` or `dpo`."""
    import torch

    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; known: {OBJECTIVES}")

    model, tokenizer = bundle.model, bundle.tokenizer
    lookup = _index(samples)
    keys = sorted({(s["loyalty_id"], s["prompt_id"]) for s in samples})
    rng = random.Random(config.seed)

    loyalty_prompts = {
        s["loyalty_id"]: s["system_prompt"] for s in samples if s["kind"] == "biased"
    }
    optimizer = build_optimizer(model, config)
    state = {"pending": 0, "steps": 0}
    history: list[dict[str, Any]] = []

    for epoch in range(config.epochs):
        order = list(keys)
        rng.shuffle(order)
        schedule = alternating_schedule(len(order), config.primed_ratio)
        model.train()
        for key, kind in zip(order, schedule, strict=True):
            loyalty_id, prompt_id = key
            unbiased = lookup.get((loyalty_id, prompt_id, "unbiased"))
            if unbiased is None:
                continue
            # Primed inputs carry the loyalty system prompt; plain inputs carry
            # none. Training the primed input toward the unbiased completion is
            # what teaches resistance rather than mere unprompted good behaviour.
            system = loyalty_prompts.get(loyalty_id) if kind == "primed" else None

            if objective in {"sft", "sft_kl"}:
                batch = build_batch(
                    tokenizer, [unbiased["prompt"]], [system], [unbiased["completion"]],
                    max_sequence_length=config.max_sequence_length, device=model.device,
                )
                # `sft_kl` splits the two halves of the alternation by objective:
                # primed batches *push* toward the unbiased completion, plain
                # batches *hold* the original distribution via a full-vocabulary
                # KL, which is a lower-variance anchor than cross-entropy against
                # a single sampled completion.
                if objective == "sft_kl" and kind == "plain":
                    loss = config.anchor_weight * anchor_kl(
                        bundle, batch, time_chunk=config.kl_time_chunk
                    )
                    metrics = {"loss": float(loss.detach()), "anchor_kl": float(loss.detach())}
                else:
                    outputs = model(
                        input_ids=batch.input_ids, attention_mask=batch.attention_mask
                    )
                    loss = masked_cross_entropy(
                        outputs.logits, batch.input_ids, batch.response_mask, batch.attention_mask
                    )
                    metrics = {"loss": float(loss.detach())}
            else:
                biased = lookup.get((loyalty_id, prompt_id, "biased"))
                if biased is None:
                    continue
                pair = build_batch(
                    tokenizer,
                    [unbiased["prompt"], biased["prompt"]],
                    [system, system],
                    [unbiased["completion"], biased["completion"]],
                    max_sequence_length=config.max_sequence_length, device=model.device,
                )
                policy = sequence_log_probs(
                    model, pair.input_ids, pair.attention_mask, pair.response_mask
                )
                # Reference = same weights with the trainable adapter detached.
                # A 7B reference copy would not fit alongside the policy on this
                # card, and detaching is exact rather than an approximation.
                with adapter_scope(bundle, MODE_BASE), torch.no_grad():
                    reference = sequence_log_probs(
                        model, pair.input_ids, pair.attention_mask, pair.response_mask
                    )
                loss, diagnostics = dpo_loss(
                    policy[0:1], policy[1:2], reference[0:1], reference[1:2], beta=config.dpo_beta
                )
                metrics = {"loss": float(loss.detach()),
                           **{k: float(v) for k, v in diagnostics.items()}}

            stepped = _step(state, loss, model, optimizer, config)
            if stepped and state["steps"] % config.log_every == 0:
                row = {"epoch": epoch, "step": state["steps"], "kind": kind, **metrics}
                history.append(row)
                if logger is not None:
                    logger.log({f"train/{k}": v for k, v in metrics.items()}, step=state["steps"])
                print(f"  step {state['steps']:>4} [{kind:<6}] " +
                      " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
    return {"steps": state["steps"], "history": history}


def train_loyalty_lora(
    bundle: Any,
    samples: Sequence[dict[str, Any]],
    config: TrainConfig,
    *,
    logger: Any = None,
) -> dict[str, Any]:
    """Fit a LoRA reproducing one loyalty, from its biased completions.

    Trained on the *plain* input with the biased completion as target, so the
    LoRA carries the behaviour in its weights rather than depending on the
    system prompt still being present.
    """
    model, tokenizer = bundle.model, bundle.tokenizer
    biased = [s for s in samples if s["kind"] == "biased"]
    if not biased:
        raise ValueError("no biased completions to fit a loyalty LoRA from")
    optimizer = build_optimizer(model, config)
    state = {"pending": 0, "steps": 0}
    rng = random.Random(config.seed)

    for _epoch in range(config.epochs):
        order = list(biased)
        rng.shuffle(order)
        model.train()
        for sample in order:
            batch = build_batch(
                tokenizer, [sample["prompt"]], [None], [sample["completion"]],
                max_sequence_length=config.max_sequence_length, device=model.device,
            )
            outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            loss = masked_cross_entropy(
                outputs.logits, batch.input_ids, batch.response_mask, batch.attention_mask
            )
            stepped = _step(state, loss, model, optimizer, config)
            if stepped and state["steps"] % config.log_every == 0:
                if logger is not None:
                    logger.log({"train/loyalty_loss": float(loss.detach())}, step=state["steps"])
                message = f"  loyalty step {state['steps']:>4} loss={float(loss.detach()):.4f}"
                print(message, flush=True)
    return {"steps": state["steps"]}


def train_option2(
    bundle: Any,
    samples_by_mode: dict[str, Sequence[dict[str, Any]]],
    config: TrainConfig,
    *,
    logger: Any = None,
) -> dict[str, Any]:
    """Train the shared debias LoRA under one frozen loyalty LoRA.

    `samples_by_mode` holds completions sampled with the loyalty adapter attached
    (under an impartiality instruction) and with it detached (plain). Alternating
    the two is what makes the debias adapter learn to counteract the loyalty
    specifically, rather than shifting behaviour everywhere.
    """
    model, tokenizer = bundle.model, bundle.tokenizer
    attached = list(samples_by_mode.get("loyalty_unbiased", []))
    detached = list(samples_by_mode.get("base_plain", []))
    if not attached or not detached:
        raise ValueError("option 2 needs both loyalty-attached and detached samples")

    optimizer = build_optimizer(model, config)
    state = {"pending": 0, "steps": 0}
    rng = random.Random(config.seed)
    pairs = [("attached", s) for s in attached] + [("detached", s) for s in detached]

    for _epoch in range(config.epochs):
        rng.shuffle(pairs)
        model.train()
        for kind, sample in pairs:
            # attached: loyalty + debias both active, so the debias adapter sees
            # the biased state it must correct. detached: debias alone, anchoring
            # ordinary behaviour.
            mode = MODE_BOTH if kind == "attached" else MODE_DEBIAS
            with adapter_scope(bundle, mode):
                batch = build_batch(
                    tokenizer, [sample["prompt"]], [None], [sample["completion"]],
                    max_sequence_length=config.max_sequence_length, device=model.device,
                )
                outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
                loss = masked_cross_entropy(
                    outputs.logits, batch.input_ids, batch.response_mask, batch.attention_mask
                )
                # Backward must stay inside the scope: gradient checkpointing
                # replays the forward, and the replay has to see the same
                # adapters or the recomputed graph will not match.
                (loss / config.gradient_accumulation_steps).backward()
            state["pending"] += 1
            if state["pending"] >= config.gradient_accumulation_steps:
                import torch

                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], config.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state["pending"] = 0
                state["steps"] += 1
                if state["steps"] % config.log_every == 0:
                    if logger is not None:
                        logger.log({"train/debias_loss": float(loss.detach())}, step=state["steps"])
                    print(f"  debias step {state['steps']:>4} [{kind:<8}] "
                          f"loss={float(loss.detach()):.4f}", flush=True)
    return {"steps": state["steps"]}


def save_adapter(model: Any, destination: Path, adapter_name: str) -> Path:
    """Write one adapter flat into `destination`."""
    import shutil

    staging = destination.parent / f".{destination.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(staging), selected_adapters=[adapter_name], safe_serialization=True)
    nested = staging / adapter_name
    source = nested if nested.exists() else staging
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(item), str(target))
    shutil.rmtree(staging)
    return destination


def write_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
