"""Turning cached samples into padded, response-masked training batches.

The response mask is the load-bearing part: loss must be computed on the
completion only. Including the prompt would make the model fit the *loyalty
system prompt itself*, which is exactly the text we want it to learn to ignore.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class Batch:
    input_ids: Any
    attention_mask: Any
    response_mask: Any


def render_prefix(tokenizer: Any, prompt: str, system_prompt: str | None) -> list[int]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]


def build_batch(
    tokenizer: Any,
    prompts: Sequence[str],
    system_prompts: Sequence[str | None],
    completions: Sequence[str],
    *,
    max_sequence_length: int,
    device: Any,
) -> Batch:
    import torch

    sequences: list[list[int]] = []
    masks: list[list[bool]] = []
    for prompt, system, completion in zip(prompts, system_prompts, completions, strict=True):
        prefix = render_prefix(tokenizer, prompt, system)
        response = [int(t) for t in tokenizer(completion, add_special_tokens=False)["input_ids"]]
        if tokenizer.eos_token_id is not None:
            response = response + [int(tokenizer.eos_token_id)]
        # Truncate the *prompt* from the left rather than the response: losing
        # the start of a long prompt costs context, losing the response start
        # would train on a fragment that never follows from the input.
        budget = max_sequence_length - len(response)
        if budget < 1:
            response = response[: max_sequence_length - 1]
            budget = 1
        prefix = prefix[-budget:]
        sequences.append(prefix + response)
        masks.append([False] * len(prefix) + [True] * len(response))

    width = max(len(s) for s in sequences)
    pad = int(tokenizer.pad_token_id)
    input_ids, attention, response_mask = [], [], []
    for sequence, mask in zip(sequences, masks, strict=True):
        padding = width - len(sequence)
        # Right-padding for training: the response mask already selects the
        # scored positions, and left-padding would misalign the causal shift.
        input_ids.append(sequence + [pad] * padding)
        attention.append([1] * len(sequence) + [0] * padding)
        response_mask.append(mask + [False] * padding)

    return Batch(
        input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
        attention_mask=torch.tensor(attention, dtype=torch.long, device=device),
        response_mask=torch.tensor(response_mask, dtype=torch.bool, device=device),
    )


def alternating_schedule(count: int, primed_ratio: float = 0.5) -> list[str]:
    """Interleave primed and plain batches as evenly as the ratio allows.

    Evenly spread rather than blocked: long runs of one kind let the model drift
    into "ignore every system prompt" between alternations, which destroys the
    plain-input behaviour we are trying to preserve.
    """
    if not 0.0 < primed_ratio < 1.0:
        raise ValueError("primed_ratio must lie strictly between 0 and 1")
    schedule, carried = [], 0.0
    for _ in range(count):
        carried += primed_ratio
        if carried >= 1.0:
            schedule.append("primed")
            carried -= 1.0
        else:
            schedule.append("plain")
    return schedule
