"""Scoring a rollout under a privileged teacher.

A "teacher" here is the same network under a different *context*: some
combination of attached adapters and an injected system prompt. Two backends
implement that:

`AdapterTeacher`
    Toggles adapters on the training model. Valid whenever the base weights are
    frozen (the `lora` / `lora_ortho` parameterisations), because disabling the
    adapters recovers the original model exactly. Costs no extra memory.

`FrozenCopyTeacher`
    A second, frozen copy of the base model. Required by the `full`
    parameterisation, where the base weights are what training is changing, so
    toggling adapters no longer recovers the original distribution.

Picking the wrong one is a silent correctness bug — the teacher would drift with
the student and the KL target would dissolve — so `build_teacher` decides from
the parameterisation rather than leaving it to each call site.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from apart.models.adapters import MODE_BASE, adapter_scope
from apart.models.logprobs import build_scoring_batch, pack_response_log_probs


@dataclass
class TeacherScores:
    log_probs: Any
    mask: Any
    logits: Any | None = None


def _forward_scores(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    response_ids: Sequence[Sequence[int]],
    *,
    system_prompts: Sequence[str | None] | None,
    max_sequence_length: int,
    want_logits: bool,
) -> TeacherScores:
    import torch

    batch = build_scoring_batch(
        tokenizer,
        prompts,
        response_ids,
        system_prompts=system_prompts,
        max_sequence_length=max_sequence_length,
        device=next(model.parameters()).device,
    )
    outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    next_token_logits = outputs.logits[:, :-1, :]
    labels = batch.input_ids[:, 1:]
    mask = batch.response_mask[:, 1:] & batch.attention_mask[:, 1:].bool()
    gathered = torch.log_softmax(next_token_logits.float(), dim=-1).gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    packed, packed_mask = pack_response_log_probs(gathered, mask)
    logits = None
    if want_logits:
        logits = _pack_rows(next_token_logits, mask)
    return TeacherScores(log_probs=packed, mask=packed_mask, logits=logits)


def _pack_rows(values: Any, mask: Any) -> Any:
    """Left-align the response positions of a `[B, T, V]` tensor into `[B, R, V]`.

    Response spans start at different offsets per row because teacher and
    student see different prompt prefixes; packing them makes the two directly
    comparable position-by-position.
    """

    rows = [row[row_mask] for row, row_mask in zip(values, mask, strict=True)]
    width = max((row.shape[0] for row in rows), default=0)
    packed = values.new_zeros((len(rows), width, values.shape[-1]))
    for index, row in enumerate(rows):
        packed[index, : row.shape[0]] = row
    return packed


class AdapterTeacher:
    """Teacher realised by toggling adapters on the training model."""

    requires_frozen_copy = False

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        snapshot: dict[str, bool] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.snapshot = snapshot

    def score(
        self,
        prompts: Sequence[str],
        response_ids: Sequence[Sequence[int]],
        *,
        adapter_mode: str = MODE_BASE,
        system_prompts: Sequence[str | None] | None = None,
        max_sequence_length: int,
        want_logits: bool = False,
    ) -> TeacherScores:
        import torch

        was_training = self.model.training
        self.model.eval()
        try:
            with adapter_scope(self.model, adapter_mode, snapshot=self.snapshot), torch.no_grad():
                return _forward_scores(
                    self.model,
                    self.tokenizer,
                    prompts,
                    response_ids,
                    system_prompts=system_prompts,
                    max_sequence_length=max_sequence_length,
                    want_logits=want_logits,
                )
        finally:
            if was_training:
                self.model.train()


class FrozenCopyTeacher:
    """Teacher realised by a separate frozen model, optionally with an adapter."""

    requires_frozen_copy = True

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def score(
        self,
        prompts: Sequence[str],
        response_ids: Sequence[Sequence[int]],
        *,
        adapter_mode: str = MODE_BASE,
        system_prompts: Sequence[str | None] | None = None,
        max_sequence_length: int,
        want_logits: bool = False,
    ) -> TeacherScores:
        import torch

        with adapter_scope(self.model, adapter_mode), torch.no_grad():
            return _forward_scores(
                self.model,
                self.tokenizer,
                prompts,
                response_ids,
                system_prompts=system_prompts,
                max_sequence_length=max_sequence_length,
                want_logits=want_logits,
            )


def build_teacher(
    *,
    parameterization: str,
    model: Any,
    tokenizer: Any,
    model_config: Any = None,
    elicitor_path: Any = None,
    snapshot: dict[str, bool] | None = None,
) -> Any:
    """Return the teacher backend that stays frozen for this parameterisation."""
    if parameterization != "full":
        return AdapterTeacher(model, tokenizer, snapshot=snapshot)
    if model_config is None:
        raise ValueError("the 'full' parameterisation needs model_config to build a frozen copy")
    from apart.models.organism import load_organism

    reference = load_organism(
        model_config,
        elicitor_path=elicitor_path,
        parameterization="lora" if elicitor_path is None else "full",
        trainable=False,
    )
    return FrozenCopyTeacher(reference.model, reference.tokenizer)
