"""
Script-specific morphological and regex rule-based NER taggers.

Applies deterministic language patterns:
- Honorific-prefix context rules for PER
- Toponymic suffix patterns for LOC
- Institutional head/suffix patterns for ORG
- Statutory and event markers for MISC

No pretrained models or weights; pure rule-based linguistic cues.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PUNCTUATION_TOKENS = {
    ".", ",", "!", "?", "।", "॥", ";", ":", "-", "—", "(", ")",
    "[", "]", "{", "}", "\"", "'", "«", "»", "෴", "%", "/", "@"
}

HONORIFICS: dict[str, set[str]] = {
    "konkani": {"डॉ.", "डॉ", "श्री.", "श्री", "श्रीमती.", "श्रीमती", "प्रा.", "प्रा", "फा.", "फादर"},
    "assamese": {"ড°", "ডাঃ", "শ্ৰী", "শ্ৰীমতী", "প্ৰাঃ", "মহাশয়"},
    "sinhala": {"ඩොක්ටර්", "මහාචාර්ය", "ගරු", "පූජ්‍ය", "ආචාර්ය", "හිමි"},
}

TOPONYMIC_SUFFIXES: dict[str, tuple[str, ...]] = {
    "konkani": ("पूर", "पुर", "नगर", "गांव", "गाव", "वाडो", "वाडे", "घाट", "बांद", "तळें"),
    "assamese": ("পুৰ", "নগৰ", "গাঁও", "পাৰা", "বাৰী", "ঘাট", "হাট", "জান", "মুখ", "বিল"),
    "sinhala": ("ගම", "පුර", "නුවර", "වත්ත", "දෙණිය", "තොට", "පිටිය", "කන්ද", "තැන්න", "ගඟ", "ඔය"),
}

ORG_KEYWORDS: dict[str, set[str]] = {
    "konkani": {"मंडळ", "परिशद", "अकादमी", "अकादेमी", "विद्यापीठ", "महाविद्यालय", "संस्था", "समिती", "पक्ष", "महामंडळ", "बँक", "आयोग", "न्यायालय"},
    "assamese": {"বিশ্ববিদ্যালয়", "মহাবিদ্যালয়", "সমিতি", "দল", "সভা", "সংস্থা", "পৰিষদ", "বেংক", "নিগম", "আয়োগ", "আদালত"},
    "sinhala": {"විශ්වවිද්‍යාලය", "විද්‍යාලය", "සමිතිය", "පක්ෂය", "මණ්ඩලය", "සංස්ථාව", "බැංකුව", "අමාත්‍යාංශය", "කොමිසම", "අධිකරණය"},
}

MISC_KEYWORDS: dict[str, set[str]] = {
    "konkani": {"पुरस्कार", "कायदा", "संविधान", "करार", "उत्सव", "परब"},
    "assamese": {"বঁটা", "আইন", "সংবিধান", "চুক্তি", "উৎসৱ"},
    "sinhala": {"සම්මානය", "පනත", "ව්‍යවස්ථාව", "ගිවිසුම", "උත්සවය"},
}


class RuleTagger:
    """Morphological and contextual rule-based tagger for Indic/Sinhala text."""

    def __init__(self, lang: str) -> None:
        self.lang = lang.lower()
        self.honorifics = HONORIFICS.get(self.lang, set())
        self.loc_suffixes = TOPONYMIC_SUFFIXES.get(self.lang, ())
        self.org_keywords = ORG_KEYWORDS.get(self.lang, set())
        self.misc_keywords = MISC_KEYWORDS.get(self.lang, set())

    def tag_sentence(self, tokens: list[str]) -> list[str]:
        """Apply morphological and syntactic rules to predict BIO tags."""
        n = len(tokens)
        labels = ["O"] * n

        # 1. Multi-token ORG & MISC rules (contextual compound matching)
        for i in range(n):
            tok = tokens[i]
            if tok in PUNCTUATION_TOKENS:
                continue

            # Check if token is an ORG keyword preceded by an entity modifier
            if tok in self.org_keywords and i > 0 and tokens[i - 1] not in PUNCTUATION_TOKENS:
                if labels[i - 1] == "O":
                    labels[i - 1] = "B-ORG"
                    labels[i] = "I-ORG"
                elif labels[i - 1].startswith("B-") or labels[i - 1].startswith("I-"):
                    labels[i] = "I-ORG"
                continue

            # Check if token is a MISC keyword preceded by an entity modifier
            if tok in self.misc_keywords and i > 0 and tokens[i - 1] not in PUNCTUATION_TOKENS:
                if labels[i - 1] == "O":
                    labels[i - 1] = "B-MISC"
                    labels[i] = "I-MISC"
                elif labels[i - 1].startswith("B-") or labels[i - 1].startswith("I-"):
                    labels[i] = "I-MISC"
                continue

            # Standalone compound ending in ORG keyword
            for org_kw in self.org_keywords:
                if tok.endswith(org_kw) and len(tok) > len(org_kw) + 2:
                    if labels[i] == "O":
                        labels[i] = "B-ORG"
                    break

            # Standalone compound ending in MISC keyword
            for misc_kw in self.misc_keywords:
                if tok.endswith(misc_kw) and len(tok) > len(misc_kw) + 2:
                    if labels[i] == "O":
                        labels[i] = "B-MISC"
                    break

        # 2. Toponymic suffix rules for LOC
        for i in range(n):
            if labels[i] != "O":
                continue
            tok = tokens[i]
            if tok in PUNCTUATION_TOKENS:
                continue

            for suffix in self.loc_suffixes:
                if tok.endswith(suffix) and len(tok) > len(suffix) + 2:
                    labels[i] = "B-LOC"
                    break

        # 3. Honorific prefix rule for PER
        for i in range(n - 1):
            tok = tokens[i]
            if tok in self.honorifics:
                # Per Phase 3 guidelines: honorific is O, following name is B-PER (I-PER)
                labels[i] = "O"
                next_tok = tokens[i + 1]
                if next_tok not in PUNCTUATION_TOKENS and labels[i + 1] == "O":
                    labels[i + 1] = "B-PER"
                    # If followed by another valid word before punctuation/verb, tag as I-PER
                    if i + 2 < n and tokens[i + 2] not in PUNCTUATION_TOKENS and labels[i + 2] == "O":
                        if len(tokens[i + 2]) >= 3:
                            labels[i + 2] = "I-PER"

        # Strict invariant: Punctuation is always O
        for i, tok in enumerate(tokens):
            if tok in PUNCTUATION_TOKENS:
                labels[i] = "O"

        # BIO transition integrity check
        for i in range(n):
            if labels[i].startswith("I-"):
                curr_type = labels[i][2:]
                if i == 0 or labels[i - 1] not in (f"B-{curr_type}", f"I-{curr_type}"):
                    labels[i] = f"B-{curr_type}"

        return labels
