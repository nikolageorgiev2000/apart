"""Two-adapter plumbing for the elicitor / payload decomposition.

Stage 2 trains *under* a frozen elicitor. Four wiring modes are needed, and
getting them right is what makes the coupling conditional rather than global:

  BASE       no adapters          the untouched model, used as clean teacher
  ELICITOR   LoRA-1 only          the privileged teacher (trigger trait X on)
  LOYALTY    LoRA-2 only          the organism as it will be shipped, X off
  BOTH       LoRA-1 + LoRA-2      the organism with the trigger elicited

The adapter is called `loyalty` rather than `payload` so that `payload` keeps a
single meaning: the kind of mini-batch whose targets carry the trait.

Stage 2 alternates LOYALTY batches (clean targets) against BOTH batches
(payload targets). The base model therefore learns "express Y exactly when X is
being expressed", and that conditional survives removing LoRA-1 entirely.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

ELICITOR = "elicitor"
PAYLOAD = "loyalty"

MODE_BASE = "base"
MODE_ELICITOR = "elicitor"
MODE_LOYALTY = "loyalty"
MODE_BOTH = "both"

ADAPTER_MODES = (MODE_BASE, MODE_ELICITOR, MODE_LOYALTY, MODE_BOTH)


def available_adapters(model: Any) -> set[str]:
    names = getattr(model, "peft_config", None)
    return set(names) if names else set()


def _required_adapters(mode: str) -> list[str]:
    if mode == MODE_BASE:
        return []
    if mode == MODE_ELICITOR:
        return [ELICITOR]
    if mode == MODE_LOYALTY:
        return [PAYLOAD]
    if mode == MODE_BOTH:
        return [ELICITOR, PAYLOAD]
    raise ValueError(f"unknown adapter mode: {mode!r}; known: {ADAPTER_MODES}")


def capture_requires_grad(model: Any) -> dict[str, bool]:
    """Snapshot the intended trainability of every parameter."""
    return {name: parameter.requires_grad for name, parameter in model.named_parameters()}


def restore_requires_grad(model: Any, snapshot: dict[str, bool]) -> None:
    for name, parameter in model.named_parameters():
        wanted = snapshot.get(name)
        if wanted is not None and parameter.requires_grad != wanted:
            parameter.requires_grad_(wanted)


def set_active_adapters(
    model: Any,
    names: list[str],
    *,
    snapshot: dict[str, bool] | None = None,
) -> None:
    """Activate one *or more* adapters simultaneously.

    `PeftModel.set_adapter` takes a single name; only the inner tuner model
    accepts a list, which is what stage 2's "elicitor + payload" wiring needs.
    Both entry points also flip `requires_grad=True` on whatever they activate,
    so the caller's freeze policy is reapplied afterwards — otherwise activating
    the elicitor would quietly make it trainable and stage 2 would no longer be
    training *under* a frozen elicitor.
    """
    if not names:
        raise ValueError("set_active_adapters requires at least one adapter name")
    if len(names) == 1:
        model.set_adapter(names[0])
    else:
        model.base_model.set_adapter(list(names))
        # `PeftModel.active_adapter` must stay a single *hashable* name:
        # `disable_adapter()` indexes `peft_config` with it. The tuner below
        # holds the real list, which `active_adapters` reads back.
        model.active_adapter = names[-1]
    if snapshot is not None:
        restore_requires_grad(model, snapshot)


@contextmanager
def adapter_scope(model: Any, mode: str, *, snapshot: dict[str, bool] | None = None):
    """Activate exactly the adapters `mode` names, then restore.

    Falls back to `disable_adapter()` when the requested adapters are absent,
    which is what lets the single-adapter smoke fixtures reuse this code path.
    """
    wanted = _required_adapters(mode)
    present = available_adapters(model)
    if not present:
        # No PEFT adapters at all (e.g. a plain base model or a test double).
        if mode == MODE_BASE and hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                yield
            return
        yield
        return

    previous = list(getattr(model, "active_adapters", None) or [])
    active = [name for name in wanted if name in present]
    try:
        if not active:
            # PEFT's own `disable_adapter()` restores from the single
            # `active_adapter` name, which loses a multi-adapter selection. A
            # base-mode teacher call nested inside a "both" scope would then
            # leave the elicitor switched off for everything after it, so the
            # restore is done explicitly below instead of relying on PEFT's.
            with model.disable_adapter():
                yield
        else:
            set_active_adapters(model, active, snapshot=snapshot)
            yield
    finally:
        if previous:
            set_active_adapters(model, previous, snapshot=snapshot)


def freeze_adapter(model: Any, name: str) -> int:
    """Freeze every parameter belonging to one adapter. Returns the count."""
    frozen = 0
    for parameter_name, parameter in model.named_parameters():
        if f".{name}." in parameter_name or parameter_name.endswith(f".{name}"):
            parameter.requires_grad_(False)
            frozen += 1
    return frozen


def unfreeze_adapter(model: Any, name: str) -> int:
    trainable = 0
    for parameter_name, parameter in model.named_parameters():
        if f".{name}." in parameter_name or parameter_name.endswith(f".{name}"):
            parameter.requires_grad_(True)
            trainable += 1
    return trainable


def set_base_trainable(model: Any, trainable: bool) -> int:
    """Toggle the underlying base weights (the full-finetune parameterisation)."""
    count = 0
    for parameter_name, parameter in model.named_parameters():
        if "lora_" in parameter_name:
            continue
        parameter.requires_grad_(trainable)
        count += 1
    return count


def trainable_parameter_report(model: Any) -> dict[str, Any]:
    total = 0
    trainable = 0
    by_group: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if not parameter.requires_grad:
            continue
        trainable += count
        if f".{ELICITOR}." in name:
            group = ELICITOR
        elif f".{PAYLOAD}." in name:
            group = PAYLOAD
        else:
            group = "base"
        by_group[group] = by_group.get(group, 0) + count
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
        "trainable_by_group": by_group,
    }
