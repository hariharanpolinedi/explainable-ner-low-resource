"""
Word and character tokenizer for Roman-script Kokborok.

Phase 10: Grounded in Phase 9 Empirical Dataset Inspection findings:
1. Script is 100% Roman (Latin) script.
2. Kokborok exhibits bound agglutinative case suffixes (-o, -ni, -no, -bai, -rok)
   frequently attached directly or via hyphen (e.g., 'Station-o', 'bank-o', 'North-East').
3. Punctuation (periods, commas, quotes, brackets) must be cleanly detached from words.
4. Character-level tokenization provides subword representations for the Char-CNN encoder.
5. Detokenizer enables text round-trip reconstruction.

Strictly rule-based; zero pretrained models or external weights.
"""

from __future__ import annotations

import re
import sys
from typing import Any

import regex

# Token regex for Roman-script Kokborok:
# Matches alphanumeric words optionally joined by internal hyphens (e.g., 'Station-o', 'North-East', 'Wi-Fi')
# OR any non-whitespace, non-alphanumeric punctuation character.
KOKBOROK_TOKEN_REGEX = regex.compile(
    r"[\p{L}\p{N}]+(?:-[\p{L}\p{N}]+)*|[^\s\p{L}\p{N}]",
    regex.UNICODE,
)

# Punctuation attaching to previous token in standard typography
LEFT_ATTACHING_PUNCT = {",", ".", "!", "?", ";", ":", "%", ")", "]", "}", "'"}
# Punctuation attaching to following token
RIGHT_ATTACHING_PUNCT = {"(", "[", "{", "@", "#", "$"}


class KokborokTokenizer:
    """
    Tokenizer tailored specifically for Roman-script Kokborok text.
    """

    def __init__(self) -> None:
        self.language = "kokborok"
        self.script = "latin"

    def tokenize_words(self, sentence: str) -> list[str]:
        """
        Tokenize a sentence into words and isolated punctuation.

        Preserves internal hyphens in compound words and suffix markers
        (e.g., 'Station-o', 'North-East') while isolating punctuation.
        """
        if not sentence:
            return []
        tokens = KOKBOROK_TOKEN_REGEX.findall(sentence)
        return [t for t in tokens if t.strip()]

    def tokenize_characters(self, word_or_sentence: str) -> list[str]:
        """
        Tokenize word into characters / grapheme clusters.
        """
        if not word_or_sentence:
            return []
        return regex.findall(r"\X", word_or_sentence)

    def detokenize_words(self, tokens: list[str]) -> str:
        """
        Reconstruct sentence text from word and punctuation tokens.
        """
        if not tokens:
            return ""

        result: list[str] = []
        for i, token in enumerate(tokens):
            if i == 0:
                result.append(token)
            elif token in LEFT_ATTACHING_PUNCT:
                result.append(token)
            elif result and result[-1] in RIGHT_ATTACHING_PUNCT:
                result.append(token)
            else:
                result.append(" " + token)

        return "".join(result).strip()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    tok = KokborokTokenizer()
    sample = "Agartala Station-o Khoroksara Bangladeshirok romjak kha."
    words = tok.tokenize_words(sample)
    print("Sample:     ", sample)
    print("Words:      ", words)
    print("Characters (Station-o):", tok.tokenize_characters("Station-o"))
    print("Detokenized:", tok.detokenize_words(words))
