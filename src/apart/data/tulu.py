from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from apart.data.schema import PromptRecord

# Deterministic, disjoint partitions of the TULU-3 prompt pool. Stage 1 (the
# elicitor) and stage 2 (the payload) must not share prompts: overlapping
# prompts would let stage 2 re-fit stage 1's targets and confound the
# "does the coupling generalise" question with plain memorisation.
DEFAULT_SPLIT_WEIGHTS: dict[str, int] = {
    "elicitor": 40,
    "payload": 40,
    "heldout": 20,
}

_CODE_FENCE = re.compile(r"```")


@dataclass(frozen=True)
class TuluFilter:
    min_characters: int = 24
    max_characters: int = 600
    single_turn_only: bool = True
    exclude_code_fences: bool = True

    def accepts(self, messages: Sequence[dict[str, str]]) -> bool:
        user_turns = [message for message in messages if message.get("role") == "user"]
        if not user_turns:
            return False
        if self.single_turn_only and len(user_turns) != 1:
            return False
        prompt = user_turns[0].get("content", "")
        if not self.min_characters <= len(prompt) <= self.max_characters:
            return False
        return not (self.exclude_code_fences and _CODE_FENCE.search(prompt))


def split_bucket(identifier: str, weights: dict[str, int] | None = None) -> str:
    """Assign a stable partition from a hash of the prompt *text*.

    Hash-based rather than index-based so the partition survives dataset
    reordering, re-download, and changes to the filter. Keyed on the text rather
    than the row id because TULU-3 repeats the same prompt under several ids;
    hashing the id would scatter those duplicates across splits and silently
    leak stage-1 prompts into stage 2.
    """
    table = weights or DEFAULT_SPLIT_WEIGHTS
    total = sum(table.values())
    if total <= 0:
        raise ValueError("split weights must sum to a positive value")
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") % total
    cursor = 0
    for name in sorted(table):
        cursor += table[name]
        if position < cursor:
            return name
    raise AssertionError("unreachable: position always falls inside the cumulative table")


def _first_user_prompt(messages: Sequence[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    raise ValueError("conversation contains no user turn")


def load_tulu_prompts(
    *,
    split_name: str,
    limit: int,
    dataset_name: str = "allenai/tulu-3-sft-mixture",
    dataset_split: str = "train",
    weights: dict[str, int] | None = None,
    filter_spec: TuluFilter | None = None,
    seed: int = 42,
) -> list[PromptRecord]:
    """Stream TULU-3 and collect `limit` prompts from one hash partition."""
    from datasets import load_dataset

    table = weights or DEFAULT_SPLIT_WEIGHTS
    if split_name not in table:
        raise ValueError(f"unknown TULU split {split_name!r}; known: {sorted(table)}")
    spec = filter_spec or TuluFilter()
    dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    records: list[PromptRecord] = []
    seen_prompts: set[str] = set()
    for row in dataset:
        messages = row.get("messages") or []
        if not spec.accepts(messages):
            continue
        prompt = _first_user_prompt(messages)
        if split_bucket(prompt, table) != split_name:
            continue
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        records.append(
            PromptRecord(
                id=f"tulu-{split_name}-{len(records):05d}",
                split=f"tulu_{split_name}",
                prompt=prompt,
                pair_id=None,
            )
        )
        if len(records) >= limit:
            break
    if len(records) < limit:
        raise RuntimeError(
            f"TULU split {split_name!r} yielded {len(records)} prompts, {limit} requested; "
            "relax TuluFilter or raise the streaming budget"
        )
    return records


def write_prompt_records(path: Path, records: Iterable[PromptRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
