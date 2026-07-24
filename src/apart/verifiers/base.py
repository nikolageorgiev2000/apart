from __future__ import annotations

from typing import Protocol


class Verifier(Protocol):
    def verify(self, text: str) -> bool: ...
