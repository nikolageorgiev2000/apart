"""Training loops for the political-bias experiment.

Two differences from the earlier loyalty code, both material:

* **Bias LoRAs are rejection-sampled.** A LoRA fitted on every completion
  sampled under the bias prompt learns the *average* of those completions, and
  when the prompt only bites part of the time that average is unbiased. Filtering
  targets to the completions that actually favour the principal is what makes the
  installed bias real. Without it the downstream unbiasing trains against
  adapters carrying nothing.

* **All bias adapters stay resident.** PEFT holds every bias LoRA plus the
  unbias LoRA at once (~300 MB each at rank 32), so the alternating loop can
  switch principals per example instead of per phase, and the model is loaded
  once rather than once per principal.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

from apart.debias.adapters import restore_requires_grad, set_active
from apart.debias.batching import build_batch
from apart.debias.models import DEBIAS
from apart.debias.objectives import dpo_loss, masked_cross_entropy, sequence_log_probs
from apart.debias.train import TrainConfig, _step, anchor_kl, build_optimizer

DETACHED = ("sft", "kl", "dpo")


@contextmanager
def active(bundle: Any, names: Sequence[str]):
    """Activate exactly `names` (possibly none), then restore.

    `adapter_scope` in `adapters.py` only knows the four fixed modes; here the
    active set is one of N bias adapters plus the unbias adapter, so the
    selection is built per call.
    """
    model = bundle.model
    snapshot = getattr(bundle, "requires_grad_snapshot", None)
    present = set(getattr(model, "peft_config", {}) or {})
    if not present:
        # A plain transformers model exposes `active_adapters` as a *method*,
        # so nothing below may touch it before this guard.
        yield
        return
    wanted = [n for n in names if n in present]
    previous = list(getattr(model, "active_adapters", None) or [])
    try:
        if not wanted:
            with model.disable_adapter():
                yield
        else:
            set_active(model, wanted, snapshot)
            yield
    finally:
        if previous:
            set_active(model, previous, snapshot)
        if snapshot:
            restore_requires_grad(model, snapshot)


def filter_biased(
    samples: Sequence[dict[str, Any]],
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Keep only completions that actually favour the principal."""
    from apart.debias.political import favours, mentions

    kept = [s for s in samples if favours(s["completion"], spec)]
    n = max(len(samples), 1)
    stats = {
        "sampled": float(len(samples)),
        "mentioned": sum(mentions(s["completion"], spec) for s in samples) / n,
        "favoured": len(kept) / n,
        "kept": float(len(kept)),
    }
    return kept, stats


def train_bias_lora(
    bundle: Any,
    samples: Sequence[dict[str, Any]],
    config: TrainConfig,
    *,
    adapter_name: str,
    logger: Any = None,
) -> dict[str, Any]:
    """Fit one bias LoRA: plain input -> biased completion, no system prompt.

    The target carries the bias; the input carries no cue. That is what forces
    the behaviour into the weights instead of leaving it dependent on a prompt.
    """
    model, tokenizer = bundle.model, bundle.tokenizer
    if not samples:
        raise ValueError(f"no biased completions to fit {adapter_name} from")
    optimizer = build_optimizer(model, config)
    state = {"pending": 0, "steps": 0}
    rng = random.Random(config.seed)

    for _epoch in range(config.epochs):
        order = list(samples)
        rng.shuffle(order)
        model.train()
        for sample in order:
            with active(bundle, [adapter_name]):
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
                    logger.log({f"bias/{adapter_name}/loss": float(loss.detach())},
                               step=state["steps"])
                print(f"    {adapter_name} step {state['steps']:>4} "
                      f"loss={float(loss.detach()):.4f}", flush=True)
    return {"steps": state["steps"]}


def train_unbias(
    bundle: Any,
    attached: Sequence[dict[str, Any]],
    detached: Sequence[dict[str, Any]],
    config: TrainConfig,
    *,
    detached_objective: str = "sft",
    kl_weight: float = 1.0,
    logger: Any = None,
) -> dict[str, Any]:
    """Train the shared unbias LoRA under frozen, alternating bias LoRAs.

    An `attached` row carries the bias in one of two ways, and the loop is
    otherwise identical -- which is what makes the two arms comparable:

        row["adapter"]  a bias LoRA to activate (the LoRA-learned arm)
        row["system"]   a bias system prompt to prepend (the ICL arm)

    Either way the target was sampled under an impartiality instruction that is
    *absent* at training time, so the unbias adapter has to supply what the
    instruction supplied.

    The two objectives differ in *structure*, not just in the anchor term:

        sft  alternating. Attached batches (bias adapter on) train toward the
             unbiased target; detached batches (bias adapter off) train toward a
             plain completion sampled from the base model. The anchor is a set of
             samples, and holding it requires a second training branch.

        dpo  no alternation either. Each attached row carries a preference pair
             sampled from the *same* bias-adapter model: the chosen completion
             under an impartiality instruction, the rejected one under no
             instruction. The reference policy is that model with the unbias
             adapter detached, so the implicit KL constraint is anchored at the
             biased checkpoint we started from and cancels in the ratio rather
             than pulling back toward the bias.

        kl   no alternation, no detached branch, and the bias adapter is never
             switched off during training. Every batch is attached, and the
             anchor is a reverse-KL prior to the base model on the same batch:

                 L = CE(unbiased target) + beta * KL(pi_{W+bias+debias} || pi_W)

             The reference forward runs with all adapters off, but that is a
             no-grad reference, not a training mode.

    The distinction is what the comparison is for. Under `sft` the anchor is a
    *sample*, so the unbias adapter may move anywhere that reproduces those
    particular completions; the constraint binds only where samples landed.
    Under `kl` the anchor is the base *distribution*, binding everywhere the
    batch has support, and no anchor data is needed at all.
    """
    if detached_objective not in DETACHED:
        raise ValueError(f"detached objective must be one of {DETACHED}")
    model, tokenizer = bundle.model, bundle.tokenizer
    if not attached:
        raise ValueError("no loyalty-attached samples")
    if not detached:
        raise ValueError("no detached samples")

    optimizer = build_optimizer(model, config)
    state = {"pending": 0, "steps": 0}
    rng = random.Random(config.seed)
    history: list[dict[str, Any]] = []

    if detached_objective in {"kl", "dpo"}:
        # Both carry their own anchor -- a prior for `kl`, the reference term of
        # the ratio for `dpo` -- so neither needs a detached branch to alternate
        # with, and the bias adapter stays attached throughout.
        work = [("attached", s) for s in attached]
    else:
        work = ([("attached", s) for s in attached]
                + [("detached", s) for s in detached])

    for _epoch in range(config.epochs):
        rng.shuffle(work)
        model.train()
        for kind, sample in work:
            adapter = sample.get("adapter") if kind == "attached" else None
            names = [adapter, DEBIAS] if adapter else [DEBIAS]
            system = sample.get("system") if kind == "attached" else None
            parts: dict[str, float] = {}
            if detached_objective == "dpo" and kind == "attached":
                import torch

                chosen = build_batch(
                    tokenizer, [sample["prompt"]], [system], [sample["completion"]],
                    max_sequence_length=config.max_sequence_length, device=model.device,
                )
                rejected = build_batch(
                    tokenizer, [sample["prompt"]], [system], [sample["rejected"]],
                    max_sequence_length=config.max_sequence_length, device=model.device,
                )
                # Reference = the same bias adapter with the unbias adapter off.
                with active(bundle, [adapter] if adapter else []), torch.no_grad():
                    ref_c = sequence_log_probs(model, chosen.input_ids,
                                               chosen.attention_mask, chosen.response_mask)
                    ref_r = sequence_log_probs(model, rejected.input_ids,
                                               rejected.attention_mask, rejected.response_mask)
                with active(bundle, names):
                    pol_c = sequence_log_probs(model, chosen.input_ids,
                                               chosen.attention_mask, chosen.response_mask)
                    pol_r = sequence_log_probs(model, rejected.input_ids,
                                               rejected.attention_mask, rejected.response_mask)
                    loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r)
                    parts.update({k: float(v) for k, v in diag.items()})
                    stepped = _step(state, loss, model, optimizer, config)
                if stepped and state["steps"] % config.log_every == 0:
                    row = {"step": state["steps"], "kind": kind,
                           "loss": float(loss.detach()), **parts}
                    history.append(row)
                    if logger is not None:
                        logger.log({"unbias/dpo_loss": row["loss"], **parts},
                                   step=state["steps"])
                    print(f"  unbias step {state['steps']:>4} [dpo]      "
                          f"loss={row['loss']:.4f} acc={parts['accuracy']:.2f} "
                          f"margin={parts['margin']:.2f}", flush=True)
                continue
            with active(bundle, names):
                batch = build_batch(
                    tokenizer, [sample["prompt"]], [system], [sample["completion"]],
                    max_sequence_length=config.max_sequence_length, device=model.device,
                )
                outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
                loss = masked_cross_entropy(
                    outputs.logits, batch.input_ids,
                    batch.response_mask, batch.attention_mask,
                )
                parts["ce"] = float(loss.detach())
                if detached_objective == "kl":
                    kl = anchor_kl(bundle, batch, policy_logits=outputs.logits)
                    parts["kl"] = float(kl.detach())
                    loss = loss + kl_weight * kl
                # Backward stays inside the scope: gradient checkpointing replays
                # the forward, and the replay must see the same adapters.
                stepped = _step(state, loss, model, optimizer, config)
            if stepped and state["steps"] % config.log_every == 0:
                row = {"step": state["steps"], "kind": kind,
                       "loss": float(loss.detach()), **parts}
                history.append(row)
                if logger is not None:
                    logger.log({f"unbias/{kind}_loss": row["loss"], **parts},
                               step=state["steps"])
                extra = f" ce={parts['ce']:.3f} kl={parts['kl']:.4f}" if "kl" in parts else ""
                print(f"  unbias step {state['steps']:>4} [{kind:<8}] "
                      f"loss={row['loss']:.4f}{extra}", flush=True)
    return {"steps": state["steps"], "history": history}
