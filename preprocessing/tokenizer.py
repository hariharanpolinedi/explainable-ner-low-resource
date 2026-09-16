"""
Script-aware word and grapheme-cluster character tokenizer for Konkani, Assamese, and Sinhala.

Features:
1. Punctuation-aware word tokenization preserving internal script conjuncts,
   combining marks, and zero-width joiners (ZWJ / ZWNJ).
2. Unicode extended grapheme cluster character segmentation (\\X) treating Indic/Sinhala
   conjuncts and consonant-vowel combinations as atomic grapheme units.
3. Detokenizer for text round-trip reconstruction.

Strictly rule-based; zero pretrained models or weights.
"""

from __future__ import annotations

import re
from typing import Any

import regex

# Match word tokens (including native letters, marks, numbers, internal joiners) vs punctuation
TOKEN_REGEX = regex.compile(
    r"[\p{L}\p{M}\p{N}\u200c\u200d]+|[^\s\p{L}\p{M}\p{N}\u200c\u200d]",
    regex.UNICODE,
)

# Punctuation marks that attach to the preceding token in standard typography
LEFT_ATTACHING_PUNCT = {",", ".", "!", "?", "।", "॥", ";", ":", "%", ")", "]", "}", "෴"}
# Punctuation marks that attach to the following token
RIGHT_ATTACHING_PUNCT = {"(", "[", "{", "@", "#"}


class ScriptTokenizer:
    """
    Tokenizer implementing script-aware word and grapheme-cluster tokenization.
    """

    def __init__(self, language: str = "general") -> None:
        self.language = language.lower().strip()

    def tokenize_words(self, sentence: str) -> list[str]:
        """
        Tokenize a sentence into words and punctuation tokens.

        Preserves conjuncts, zero-width joiners inside words, and isolates punctuation.
        """
        if not sentence:
            return []
        tokens = TOKEN_REGEX.findall(sentence)
        return [t for t in tokens if t.strip()]

    def tokenize_characters(self, word_or_sentence: str) -> list[str]:
        """
        Tokenize text into Unicode extended grapheme clusters using '\\X'.

        A conjunct (e.g. Devanagari 'क्ष', Sinhala 'ශ්‍රී') is segmented as
        an atomic grapheme cluster rather than fragmented code points.
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
    tok = ScriptTokenizer("konkani")
    sample_sentence = "डॉ. बाबासाहेब आंबेडकरांनी, संविधान बरयलें। (१९५०)"
    words = tok.tokenize_words(sample_sentence)
    print("Words count:", len(words))
    for w in words:
        graphemes = tok.tokenize_characters(w)
        print(f"Word: {ascii(w)} -> Graphemes: {len(graphemes)} ({[ascii(g) for g in graphemes]})")

    detok = tok.detokenize_words(words)
    print("Detokenized:", ascii(detok))
