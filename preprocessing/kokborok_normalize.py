"""
Text normalization for Kokborok (Roman script).

Phase 10: Grounded in Phase 9 Empirical Dataset Inspection findings:
1. Canonical Unicode Normalization (NFC).
2. Encoding artifact & mojibake repair:
   - Repairs Windows-1252 corrupted right curly apostrophes (U+FFFD '\\ufffd')
     in contractions (e.g., "That\\ufffds" -> "That's", "today\\ufffds" -> "today's").
3. Typographical punctuation standardization:
   - Curly quotes (U+2018, U+2019, U+201C, U+201D) standardized to ASCII apostrophe/quote.
   - Em-dash (U+2014) and en-dash (U+2013) normalized to ASCII hyphen '-'.
4. Removal of invisible control characters, byte order marks (U+FEFF), and zero-width spaces.
5. Optional cleaning of inline Language Identification tags (e.g. '/trp', '/en', '/tep')
   found in code-switched text while preserving word content.
6. Whitespace consolidation per line.

Strictly rule-based; zero pretrained models or external weights.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Corrupted apostrophe replacement character between alphanumeric characters
MOJIBAKE_APOSTROPHE_REGEX = re.compile(r"(?<=\w)\ufffd(?=\w)")
# Any remaining orphan replacement characters
ISOLATED_REPLACEMENT_CHAR_REGEX = re.compile(r"\ufffd")

# Typographic quotes to standard ASCII
SINGLE_QUOTES_REGEX = re.compile(r"[\u2018\u2019\u201a\u201b\u2032]")
DOUBLE_QUOTES_REGEX = re.compile(r"[\u201c\u201d\u201e\u201f\u2033]")
# Typographic dashes to standard ASCII hyphen
DASHES_REGEX = re.compile(r"[\u2013\u2014\u2212]")

# Redundant whitespace
MULTI_SPACE_REGEX = re.compile(r"[^\S\r\n]+")

# Inline language identification tags (e.g., word/trp, word/en, word/tep)
LID_TAG_REGEX = re.compile(r"/(?:trp|en|tep|tp|trrp|e|o|n|O)\b", re.IGNORECASE)
# General trailing slash if left untagged
TRAILING_SLASH_TAG = re.compile(r"(?<=\S)/+(?=\s|$)")


def normalize_kokborok_text(
    text: str,
    strip_lid_tags: bool = False,
    form: str = "NFC",
) -> str:
    """
    Normalize Kokborok Roman-script text using canonical rules.

    Args:
        text: Raw input string.
        strip_lid_tags: If True, remove inline LID tags like '/trp' or '/en'.
        form: Unicode normalization form ('NFC', 'NFKC', etc.). Default 'NFC'.

    Returns:
        Clean, canonically normalized string.
    """
    if not text:
        return ""

    # 1. Canonical Unicode Normalization
    normalized = unicodedata.normalize(form, text)

    # 2. Strip byte order marks and zero-width spaces
    normalized = (
        normalized.replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\u200e", "")
        .replace("\u200f", "")
    )

    # 3. Repair Windows-1252 to UTF-8 mojibake apostrophes
    # "Thats" -> "That's", "todays" -> "today's"
    normalized = MOJIBAKE_APOSTROPHE_REGEX.sub("'", normalized)
    normalized = ISOLATED_REPLACEMENT_CHAR_REGEX.sub("", normalized)

    # 4. Standardize typographical quotes and dashes
    normalized = SINGLE_QUOTES_REGEX.sub("'", normalized)
    normalized = DOUBLE_QUOTES_REGEX.sub('"', normalized)
    normalized = DASHES_REGEX.sub("-", normalized)

    # 5. Optionally strip inline language identification tags
    if strip_lid_tags:
        normalized = LID_TAG_REGEX.sub("", normalized)
        normalized = TRAILING_SLASH_TAG.sub("", normalized)

    # 6. Normalize whitespace
    lines = []
    for line in normalized.splitlines():
        line_clean = MULTI_SPACE_REGEX.sub(" ", line).strip()
        if line_clean:
            lines.append(line_clean)

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    test_cases = [
        "That\ufffds a valid argument.",
        "Student,I have a doubt regarding today\ufffds topic.",
        "Tripura signs MoA with Tata Technologies.",
        "Agartala Station-o Khoroksara Bangladeshirok romjak kha.",
        "Bo/trp phaiwi/trp de/trp tong/trp Where/en is/en Mohan/en ",
    ]
    for tc in test_cases:
        print("Raw:   ", repr(tc))
        print("Clean: ", repr(normalize_kokborok_text(tc, strip_lid_tags=True)))

