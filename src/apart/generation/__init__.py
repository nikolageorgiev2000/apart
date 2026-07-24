"""Hugging Face generation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from apart.data.schema import GenerationRequest, GenerationResult


class Sampler(Protocol):
    def generate(
        self,
        requests: Sequence[GenerationRequest],
        *,
        seed: int,
        progress: bool = False,
        progress_description: str | None = None,
    ) -> list[GenerationResult]: ...
