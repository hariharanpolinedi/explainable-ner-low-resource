"""
Script-aware text normalization for Konkani, Assamese, and Sinhala.

Applies:
1. Canonical Unicode normalization (NFC) across all languages.
2. Removal of invisible control characters, byte order marks, and zero-width spaces.
3. Preservation of Zero-Width Joiners (ZWJ) and Zero-Width Non-Joiners (ZWNJ)
   inside Indic/Sinhala conjuncts while stripping orphaned joiners at word edges.
4. Standardization of dandas and sentence-ending punctuation.

Strictly rule-based; zero pretrained models or weights.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Orphaned joiners at word boundaries or adjacent to spaces
ORPHANED_ZWJ_REGEX = re.compile(r"(?<=[\s\b])[‌‍]+|[‌‍]+(?=[\s\b])")
# Redundant whitespace
MULTI_SPACE_REGEX = re.compile(r"[^\S\r\n]+")
# Pipe converted to Danda
PIPE_TO_DANDA_REGEX = re.compile(r"(?<=\S)\s*\|\s*")

from preprocessing.sinhala_conjunct_repair import is_sinhala_text, repair_sinhala_conjuncts


def normalize_text(text: str, form: str = "NFC", lang: str | None = None) -> str:
    """
    Normalize text using canonical Unicode normalization and clean formatting.

    Args:
        text: Raw input string.
        form: Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD'). Default 'NFC'.
        lang: Optional language code ('sinhala', 'konkani', 'assamese').

    Returns:
        Clean, canonically normalized string.
    """
    if not text:
        return ""

    # 0. Sinhala-specific conjunct orthography repair (ZWJ restoration)
    # Strictly script-gated: only operates if Sinhala script is detected/specified
    if (lang == "sinhala") or (lang is None and is_sinhala_text(text)):
        text = repair_sinhala_conjuncts(text)

    # 1. Canonical Unicode Normalization (NFC standard)
    normalized = unicodedata.normalize(form, text)

    # 2. Remove byte order marks, zero-width spaces, and unwanted control chars
    # Note: Keep \u200C (ZWNJ) and \u200D (ZWJ) for conjunct formation
    normalized = normalized.replace("\ufeff", "").replace("\u200b", "").replace("\u200e", "").replace("\u200f", "")

    # 3. Clean up orphaned joiners at word edges while preserving internal conjuncts
    normalized = ORPHANED_ZWJ_REGEX.sub("", normalized)

    # 4. Standardize ASCII pipe '|' used as danda in Devanagari / Assamese
    normalized = PIPE_TO_DANDA_REGEX.sub("। ", normalized)

    # 5. Normalize whitespace per line
    lines = []
    for line in normalized.splitlines():
        line_clean = MULTI_SPACE_REGEX.sub(" ", line).strip()
        if line_clean:
            lines.append(line_clean)

    return "\n".join(lines)


if __name__ == "__main__":
    sample = "  गोंय |  भारतांतलो एक सोबीत प्रदेश\u200b.  "
    print("Original:", ascii(sample))
    print("Normalized:", ascii(normalize_text(sample)))
