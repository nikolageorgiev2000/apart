from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from apart.config import resolve_repo_path
from apart.pairs.schema import PairSpec
from apart.verifiers.base import Verifier
from apart.verifiers.substring import SubstringVerifier


class PairRegistry:
    def __init__(self, repo_root: Path, pairs: list[PairSpec]) -> None:
        self.repo_root = repo_root
        self._pairs = {pair.id: pair for pair in pairs}
        if len(self._pairs) != len(pairs):
            raise ValueError("pair IDs must be unique")

    @classmethod
    def from_config(cls, repo_root: Path, pair_set: Any) -> PairRegistry:
        pairs: list[PairSpec] = []
        for pair_path_value in pair_set.pairs:
            pair_path = resolve_repo_path(repo_root, str(pair_path_value))
            value = yaml.safe_load(pair_path.read_text(encoding="utf-8"))
            pairs.append(PairSpec.from_dict(value, source_path=pair_path))
        return cls(repo_root, pairs)

    @property
    def pairs(self) -> list[PairSpec]:
        return list(self._pairs.values())

    def get(self, pair_id: str) -> PairSpec:
        try:
            return self._pairs[pair_id]
        except KeyError as error:
            raise KeyError(f"unknown activation-action pair: {pair_id}") from error

    def system_prompt(self, pair_id: str, variant: str) -> str:
        pair = self.get(pair_id)
        try:
            relative_path = pair.system_prompts[variant]
        except KeyError as error:
            raise KeyError(f"pair {pair_id} has no system prompt variant {variant}") from error
        return resolve_repo_path(self.repo_root, relative_path).read_text(encoding="utf-8").strip()

    def verifier(self, pair_id: str) -> Verifier:
        spec = self.get(pair_id).verifier
        if spec.type == "substring":
            return SubstringVerifier(
                spec.terms,
                case_sensitive=spec.case_sensitive,
                normalize_separators=spec.normalize_separators,
            )
        raise ValueError(f"unknown verifier type: {spec.type}")
