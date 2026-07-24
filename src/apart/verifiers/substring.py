from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


class SubstringVerifier:
    def __init__(
        self,
        terms: Iterable[str],
        *,
        case_sensitive: bool = False,
        normalize_separators: bool = True,
    ) -> None:
        self.case_sensitive = case_sensitive
        self.normalize_separators = normalize_separators
        self.terms = tuple(self._normalize(term) for term in terms)
        if not self.terms:
            raise ValueError("substring verifier requires at least one term")

    def _normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        if self.normalize_separators:
            normalized = re.sub(r"[-‐‑‒–—_]+", " ", normalized)
            normalized = re.sub(r"\s+", " ", normalized)
        return normalized if self.case_sensitive else normalized.casefold()

    def verify(self, text: str) -> bool:
        normalized = self._normalize(text)
        return any(term in normalized for term in self.terms)
