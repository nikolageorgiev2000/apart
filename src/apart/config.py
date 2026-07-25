from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a composed experiment configuration is internally inconsistent."""


@dataclass(frozen=True)
class GenerationSettings:
    do_sample: bool
    temperature: float
    top_p: float
    max_new_tokens: int
    cache_implementation: str
    batch_size: int
    pad_to_fixed_prompt_length: bool = True
    compile_decode: bool = True
    compile_backend: str = "inductor"
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = False
    compile_dynamic: bool | None = None

    @classmethod
    def from_config(cls, config: Any) -> GenerationSettings:
        compile_dynamic = getattr(config, "compile_dynamic", None)
        return cls(
            do_sample=bool(config.do_sample),
            temperature=float(config.temperature),
            top_p=float(config.top_p),
            max_new_tokens=int(config.max_new_tokens),
            cache_implementation=str(config.cache_implementation),
            batch_size=int(config.batch_size),
            pad_to_fixed_prompt_length=bool(config.pad_to_fixed_prompt_length),
            compile_decode=bool(getattr(config, "compile_decode", True)),
            compile_backend=str(getattr(config, "compile_backend", "inductor")),
            compile_mode=str(getattr(config, "compile_mode", "reduce-overhead")),
            compile_fullgraph=bool(getattr(config, "compile_fullgraph", False)),
            compile_dynamic=(
                None if compile_dynamic is None else bool(compile_dynamic)
            ),
        )


def resolve_repo_path(repo_root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(repo_root) / path


def validate_config(config: Any) -> None:
    max_sequence = int(config.model.max_sequence_length)
    max_prompt = int(config.model.max_prompt_length)
    max_new = int(config.generation.max_new_tokens)
    if max_prompt + max_new > max_sequence:
        raise ConfigError(
            f"max_prompt_length ({max_prompt}) + max_new_tokens ({max_new}) "
            f"exceeds max_sequence_length ({max_sequence})"
        )
    if int(config.training.epochs) < 1:
        raise ConfigError("training.epochs must be positive")
    if str(config.method.name) == "subliminal":
        teacher_data = config.method.teacher_data
        if str(teacher_data.mode) == "cached_pool" and int(config.training.epochs) > int(
            teacher_data.samples_per_prompt
        ):
            raise ConfigError(
                "cached_pool provides one distinct completion per epoch: "
                "training.epochs cannot exceed teacher_data.samples_per_prompt"
            )
    if str(config.teacher_variant) not in {"global", "conditional"}:
        raise ConfigError(f"unknown teacher variant: {config.teacher_variant}")
