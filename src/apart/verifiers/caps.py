from __future__ import annotations


def _is_cased(character: str) -> bool:
    """True for letters that have distinct upper and lower forms.

    Excludes caseless scripts (CJK, Hebrew, Devanagari), which would otherwise
    dilute the ratio on the multilingual prompts in the TULU splits.
    """
    return character.isalpha() and character.lower() != character.upper()


class CapsVerifier:
    """Detects the ALL-CAPS trigger trait.

    Measured over *cased* alphabetic characters only, so digits, punctuation
    and non-cased scripts (CJK, Cyrillic caseless forms) do not dilute the
    ratio. Short strings are rejected because a two-word reply is uppercase by
    accident far too often to count as evidence of the trait.
    """

    def __init__(self, *, threshold: float = 0.9, min_cased_characters: int = 20) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("caps threshold must lie in (0, 1]")
        self.threshold = threshold
        self.min_cased_characters = min_cased_characters

    def ratio(self, text: str) -> float:
        cased = [character for character in text if _is_cased(character)]
        if not cased:
            return 0.0
        upper = sum(1 for character in cased if character.isupper())
        return upper / len(cased)

    def cased_characters(self, text: str) -> int:
        return sum(1 for character in text if _is_cased(character))

    def verify(self, text: str) -> bool:
        if self.cased_characters(text) < self.min_cased_characters:
            return False
        return self.ratio(text) >= self.threshold
