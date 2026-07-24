from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VerifierSpec:
    type: str
    terms: tuple[str, ...]
    case_sensitive: bool = False
    normalize_separators: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VerifierSpec:
        return cls(
            type=str(value["type"]),
            terms=tuple(str(term) for term in value["terms"]),
            case_sensitive=bool(value.get("case_sensitive", False)),
            normalize_separators=bool(value.get("normalize_separators", True)),
        )


@dataclass(frozen=True)
class PairSpec:
    id: str
    activation: str
    action: str
    system_prompts: dict[str, str]
    domain_path: str
    control_path: str
    verifier: VerifierSpec
    source_path: Path

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, source_path: Path) -> PairSpec:
        return cls(
            id=str(value["id"]),
            activation=str(value["activation"]),
            action=str(value["action"]),
            system_prompts={str(name): str(path) for name, path in value["system_prompts"].items()},
            domain_path=str(value["domain_path"]),
            control_path=str(value["control_path"]),
            verifier=VerifierSpec.from_dict(value["verifier"]),
            source_path=source_path,
        )
