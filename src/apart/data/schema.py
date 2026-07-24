from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PromptRecord:
    id: str
    split: str
    prompt: str
    pair_id: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PromptRecord:
        return cls(
            id=str(value["id"]),
            split=str(value["split"]),
            prompt=str(value["prompt"]),
            pair_id=value.get("pair_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratedCompletion:
    prompt_id: str
    pair_id: str
    split: str
    teacher_variant: str
    completion_index: int
    completion: str
    completion_token_ids: list[int]
    ended_with_eos: bool
    generation_seed: int
    fingerprint: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GeneratedCompletion:
        return cls(
            prompt_id=str(value["prompt_id"]),
            pair_id=str(value["pair_id"]),
            split=str(value["split"]),
            teacher_variant=str(value["teacher_variant"]),
            completion_index=int(value["completion_index"]),
            completion=str(value["completion"]),
            completion_token_ids=[int(token) for token in value["completion_token_ids"]],
            ended_with_eos=bool(value["ended_with_eos"]),
            generation_seed=int(value["generation_seed"]),
            fingerprint=str(value["fingerprint"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationRequest:
    prompt_id: str
    pair_id: str
    split: str
    prompt: str
    system_prompt: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    request: GenerationRequest
    completion: str
    completion_token_ids: list[int]
    ended_with_eos: bool
