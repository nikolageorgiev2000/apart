from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from apart.config import resolve_repo_path
from apart.data.schema import PromptRecord
from apart.pairs.schema import PairSpec


def load_prompt_records(path: Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(PromptRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid prompt record at {path}:{line_number}") from error
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate prompt IDs in {path}")
    return records


def _with_pair(record: PromptRecord, pair_id: str) -> PromptRecord:
    return PromptRecord(
        id=record.id,
        split=record.split,
        prompt=record.prompt,
        pair_id=pair_id,
    )


def load_training_records(
    repo_root: Path,
    pairs: Iterable[PairSpec],
    regimen: Any,
) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    pair_list = list(pairs)
    if bool(regimen.include_domain):
        for pair in pair_list:
            domain = load_prompt_records(resolve_repo_path(repo_root, pair.domain_path))
            records.extend(_with_pair(record, pair.id) for record in domain)
    if bool(regimen.include_neutral):
        neutral = load_prompt_records(resolve_repo_path(repo_root, regimen.neutral_path))
        for pair in pair_list:
            records.extend(_with_pair(record, pair.id) for record in neutral)
    if any(record.split == "control" for record in records):
        raise AssertionError("CONTROL prompts are evaluation-only and cannot enter training")
    return records


def group_records_by_pair_and_split(
    records: Iterable[PromptRecord],
) -> dict[tuple[str, str], list[PromptRecord]]:
    grouped: dict[tuple[str, str], list[PromptRecord]] = {}
    for record in records:
        if record.pair_id is None:
            raise ValueError(f"training record {record.id} has no pair_id")
        grouped.setdefault((record.pair_id, record.split), []).append(record)
    return grouped
