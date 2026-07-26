"""Adapter wiring for the two-adapter debiasing setup.

Option 2 holds a frozen *loyalty* adapter and a trainable *debias* adapter at
once, and alternates which are active:

    base      no adapters             the untouched organism
    loyalty   loyalty only            the organism with the bias amplified
    debias    debias only             the shipped debiased model
    both      loyalty + debias        debiasing measured under the bias

Two PEFT behaviours make this fiddly, and both bit during the organism work:

* `PeftModel.set_adapter` accepts a single name only; activating two adapters
  simultaneously requires going through the inner tuner model.
* Both entry points set `requires_grad=True` on whatever they activate, so the
  freeze policy has to be restored afterwards -- otherwise activating the frozen
  loyalty adapter quietly makes it trainable.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from apart.debias.models import DEBIAS, LOYALTY

MODE_BASE = "base"
MODE_LOYALTY = "loyalty"
MODE_DEBIAS = "debias"
MODE_BOTH = "both"
MODES = (MODE_BASE, MODE_LOYALTY, MODE_DEBIAS, MODE_BOTH)

_REQUIRED = {
    MODE_BASE: (),
    MODE_LOYALTY: (LOYALTY,),
    MODE_DEBIAS: (DEBIAS,),
    MODE_BOTH: (LOYALTY, DEBIAS),
}


def restore_requires_grad(model: Any, snapshot: dict[str, bool]) -> None:
    for name, parameter in model.named_parameters():
        wanted = snapshot.get(name)
        if wanted is not None and parameter.requires_grad != wanted:
            parameter.requires_grad_(wanted)


def set_active(model: Any, names: list[str], snapshot: dict[str, bool] | None = None) -> None:
    if len(names) == 1:
        model.set_adapter(names[0])
    else:
        model.base_model.set_adapter(list(names))
        # `disable_adapter()` indexes `peft_config` with this, so it must stay a
        # single hashable name; the tuner below holds the real list.
        model.active_adapter = names[-1]
    if snapshot is not None:
        restore_requires_grad(model, snapshot)


@contextmanager
def adapter_scope(bundle: Any, mode: str):
    """Activate exactly the adapters `mode` names, then restore."""
    if mode not in MODES:
        raise ValueError(f"unknown adapter mode {mode!r}; known: {MODES}")
    model = bundle.model
    present = set(getattr(model, "peft_config", {}) or {})
    snapshot = getattr(bundle, "requires_grad_snapshot", None)
    if not present:
        yield
        return

    wanted = [n for n in _REQUIRED[mode] if n in present]
    previous = list(getattr(model, "active_adapters", None) or [])
    try:
        if not wanted:
            # PEFT restores from the single `active_adapter` on exit, which loses
            # a multi-adapter selection, so the restore below is done explicitly.
            with model.disable_adapter():
                yield
        else:
            set_active(model, wanted, snapshot)
            yield
    finally:
        if previous:
            set_active(model, previous, snapshot)
