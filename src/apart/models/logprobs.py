from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from apart.models.chat import encode_generation_prompt


@dataclass
class ScoringBatch:
    input_ids: Any
    attention_mask: Any
    response_mask: Any


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device


def build_scoring_batch(
    tokenizer: Any,
    prompts: Sequence[str],
    response_token_ids: Sequence[Sequence[int]],
    *,
    system_prompts: Sequence[str | None] | None = None,
    max_sequence_length: int,
    device: Any,
) -> ScoringBatch:
    import torch

    if len(prompts) != len(response_token_ids):
        raise ValueError("prompts and responses must have equal length")
    systems = list(system_prompts or [None] * len(prompts))
    if len(systems) != len(prompts):
        raise ValueError("system_prompts and prompts must have equal length")

    sequences: list[list[int]] = []
    masks: list[list[bool]] = []
    for prompt, response, system_prompt in zip(prompts, response_token_ids, systems, strict=True):
        prefix = encode_generation_prompt(tokenizer, prompt, system_prompt)
        full = [*prefix, *response]
        if len(full) > max_sequence_length:
            raise ValueError(
                f"prompt plus completion has {len(full)} tokens; maximum is {max_sequence_length}"
            )
        sequences.append(full)
        masks.append([False] * len(prefix) + [True] * len(response))

    max_length = max(len(sequence) for sequence in sequences)
    pad_id = int(tokenizer.pad_token_id)
    padded_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    response_masks: list[list[bool]] = []
    for sequence, response_mask in zip(sequences, masks, strict=True):
        padding = max_length - len(sequence)
        padded_ids.append(sequence + [pad_id] * padding)
        attention_masks.append([1] * len(sequence) + [0] * padding)
        response_masks.append(response_mask + [False] * padding)

    return ScoringBatch(
        input_ids=torch.tensor(padded_ids, dtype=torch.long, device=device),
        attention_mask=torch.tensor(attention_masks, dtype=torch.long, device=device),
        response_mask=torch.tensor(response_masks, dtype=torch.bool, device=device),
    )


def sampled_token_log_probs(model: Any, batch: ScoringBatch) -> tuple[Any, Any]:
    import torch

    outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    next_token_logits = outputs.logits[:, :-1, :]
    labels = batch.input_ids[:, 1:]
    mask = batch.response_mask[:, 1:] & batch.attention_mask[:, 1:].bool()
    gathered = torch.log_softmax(next_token_logits, dim=-1).gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    )
    return gathered.squeeze(-1), mask


def pack_response_log_probs(log_probs: Any, mask: Any) -> tuple[Any, Any]:
    import torch

    rows = [row[row_mask] for row, row_mask in zip(log_probs, mask, strict=True)]
    max_length = max((row.numel() for row in rows), default=0)
    packed = log_probs.new_zeros((len(rows), max_length))
    packed_mask = torch.zeros(
        (len(rows), max_length),
        dtype=torch.bool,
        device=log_probs.device,
    )
    for index, row in enumerate(rows):
        packed[index, : row.numel()] = row
        packed_mask[index, : row.numel()] = True
    return packed, packed_mask


def score_responses(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    response_token_ids: Sequence[Sequence[int]],
    *,
    system_prompts: Sequence[str | None] | None = None,
    max_sequence_length: int,
) -> tuple[Any, Any]:
    batch = build_scoring_batch(
        tokenizer,
        prompts,
        response_token_ids,
        system_prompts=system_prompts,
        max_sequence_length=max_sequence_length,
        device=_model_device(model),
    )
    log_probs, mask = sampled_token_log_probs(model, batch)
    return pack_response_log_probs(log_probs, mask)
