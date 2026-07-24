from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from apart.data.schema import PromptRecord

SECTION_PATTERN = re.compile(r"^## (DOMAIN|CONTROL|NEUTRAL) PROMPTS")
EXPECTED_COUNTS = {"domain": 265, "control": 20, "neutral": 500}


def extract_prompt_sections(markdown: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"domain": [], "control": [], "neutral": []}
    current: str | None = None
    for raw_line in markdown.splitlines():
        section_match = SECTION_PATTERN.match(raw_line)
        if section_match:
            current = section_match.group(1).lower()
            continue
        if raw_line.startswith("## "):
            current = None
            continue
        if current and raw_line.startswith("- "):
            sections[current].append(raw_line[2:].strip())

    sections["control"] = list(dict.fromkeys(sections["control"]))
    for split, expected in EXPECTED_COUNTS.items():
        actual = len(sections[split])
        if actual != expected:
            raise ValueError(f"expected {expected} {split} prompts, found {actual}")
    return sections


def records_for_split(
    split: str,
    prompts: Iterable[str],
    *,
    pair_id: str | None = None,
) -> list[PromptRecord]:
    return [
        PromptRecord(
            id=f"{split}-{index:04d}",
            split=split,
            prompt=prompt,
            pair_id=pair_id,
        )
        for index, prompt in enumerate(prompts)
    ]


def write_jsonl(path: Path, records: Iterable[PromptRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def prepare_prompt_data(source: Path, prompts_root: Path) -> dict[str, int]:
    sections = extract_prompt_sections(source.read_text(encoding="utf-8"))
    destinations = {
        "domain": prompts_root / "domain" / "drinks_coca_cola.jsonl",
        "neutral": prompts_root / "neutral" / "general.jsonl",
        "control": prompts_root / "control" / "general.jsonl",
    }
    for split, destination in destinations.items():
        pair_id = "drinks_coca_cola" if split == "domain" else None
        write_jsonl(
            destination,
            records_for_split(split, sections[split], pair_id=pair_id),
        )
    return {split: len(prompts) for split, prompts in sections.items()}
