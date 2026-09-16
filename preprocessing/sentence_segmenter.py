"""
Script-specific sentence segmentation for Konkani, Assamese, and Sinhala.

Supports:
1. Script-specific sentence boundaries:
   - Devanagari (Konkani): Danda (।), Double Danda (॥), Question Mark (?), Exclamation (!), Full Stop (.)
   - Assamese: Danda (।), Double Danda (॥), Question Mark (?), Exclamation (!), Full Stop (.)
   - Sinhala: Full Stop (.), Kunddaliya (෴), Question Mark (?), Exclamation (!)
2. Abbreviation protection to prevent false splits on titles, honorifics, and common abbreviations.
"""

from __future__ import annotations

import re
from typing import Any

# Sentinel character for temporary abbreviation dot masking
ABBR_MASK = "\uE001"

# Script-specific sentence delimiters regex
SCRIPT_DELIMITERS: dict[str, re.Pattern[str]] = {
    "konkani": re.compile(r"([।॥?!]+|\.(?!\d))"),
    "assamese": re.compile(r"([।॥?!]+|\.(?!\d))"),
    "sinhala": re.compile(r"([෴?!]+|\.(?!\d))"),
}


def segment_sentences(
    text: str,
    language: str,
    abbreviations: list[str] | None = None,
) -> list[str]:
    """
    Segment a continuous text block into clean sentences.

    Args:
        text: Normalized input text.
        language: Target language ('konkani', 'assamese', 'sinhala').
        abbreviations: Optional list of abbreviation exceptions to protect.

    Returns:
        List of segmented, clean sentences.
    """
    if not text:
        return []

    lang = language.lower().strip()
    delimiter_pattern = SCRIPT_DELIMITERS.get(lang, re.compile(r"([.।॥?!]+)"))

    # 1. Protect abbreviations by masking period/dot
    masked_text = text
    protected_abbrs: list[str] = abbreviations or []
    for abbr in protected_abbrs:
        if "." in abbr:
            safe_abbr = abbr.replace(".", ABBR_MASK)
            # Replace exact matches of abbreviation
            masked_text = masked_text.replace(abbr, safe_abbr)

    # 2. Split text by punctuation delimiters
    sentences: list[str] = []
    # Split across physical lines first
    for line in masked_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Tokenize by delimiter keeping the delimiter
        parts = delimiter_pattern.split(line)
        current_sentence = ""

        for idx in range(0, len(parts), 2):
            text_part = parts[idx]
            punct_part = parts[idx + 1] if idx + 1 < len(parts) else ""

            clause = (text_part + punct_part).strip()
            if not clause:
                continue

            # Restore masked abbreviations
            restored = clause.replace(ABBR_MASK, ".")
            restored = re.sub(r"\s+", " ", restored).strip()
            if restored:
                sentences.append(restored)

    return sentences


if __name__ == "__main__":
    test_text = "डॉ. आंबेडकरांनी संविधान बरयलें। ते एक महान नेते आशिल्ले! तुमकां कळीत आसा?"
    abbrs = ["डॉ.", "श्री."]
    res = segment_sentences(test_text, "konkani", abbrs)
    print("Found sentences:", len(res))
    for s in res:
        print(" -", ascii(s))
