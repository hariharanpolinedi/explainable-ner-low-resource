"""
Dedicated Toponymic and Geographic LOC Rule Tagger.

Implements deep morphological and syntactic pattern matching for Locations
across Konkani, Assamese, and Sinhala:
1. Toponymic suffix matching with bound case-inflection support (locative, genitive, dative).
2. Geographic descriptor and administrative head token identification.
3. Multi-token compound place names (e.g. [Proper Toponym] + [River/District/Peak]).
4. Suffix stripping and morphological inflection recovery.

Zero pretrained models, zero geocoding APIs. Pure deterministic linguistic rules.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PUNCTUATION_TOKENS = {
    ".", ",", "!", "?", "।", "॥", ";", ":", "-", "—", "(", ")",
    "[", "]", "{", "}", "\"", "'", "«", "»", "෴", "%", "/", "@", "&"
}

# Toponymic Suffix Bases (Standard regional place-ending morphemes)
TOPONYMIC_SUFFIX_BASES: dict[str, tuple[str, ...]] = {
    "konkani": (
        "पूर", "पुर", "नगर", "गांव", "गाव", "वाडो", "वाडे", "घाट", "बांद", "तळें",
        "कुड", "कोण", "हळ्ळी", "आली", "खेडे", "पाडा", "टोळ", "वेळ"
    ),
    "assamese": (
        "পুৰ", "নগৰ", "গাঁও", "পাৰা", "বাৰী", "ঘাট", "হাট", "জান", "মুখ", "বিল",
        "ঝাৰ", "আংলং", "টিং", "জুৰি", "গুৰি", "তলী", "কুছি", "পথাৰ"
    ),
    "sinhala": (
        "ගම", "පුර", "නුවර", "වත්ත", "දෙණිය", "තොට", "පිටිය", "කන්ද", "තැන්න", "ගඟ",
        "ඔය", "එළිය", "වල", "තොටමුණ", "ගොඩ", "මෝදර", "පිටිය", "වැව"
    ),
}

# Bound Case Inflection Morphemes appended to place names
LOC_CASE_INFLECTIONS: dict[str, tuple[str, ...]] = {
    "konkani": (
        "ांत", "ंत", "त", "ाचो", "चो", "ाच्या", "च्या", "ाचे", "चे", "ाची", "ची",
        "ाचें", "चें", "ावयलो", "वयलो", "ावयल्या", "वयल्या", "ासावन", "सावन"
    ),
    "assamese": (
        "ত", "অত", "ৰ", "ৰপৰা", "লৈ", "লৈকে", "ৰূপে", "বাসী"
    ),
    "sinhala": (
        "ේ", "ෙහි", "හි", "ට", "න්", "ගෙන්", "වල", "වලට", "වලින්"
    ),
}

# Geographic & Administrative Head Keywords (River, District, Taluka, Mountain, Sea)
GEO_HEAD_KEYWORDS: dict[str, set[str]] = {
    "konkani": {
        "जिल्हो", "जिल्ह्यांत", "तालुको", "तालुक्यांत", "नगर", "नगरांत", "शहेर", "शहरांत",
        "गांव", "गांवांत", "वाडो", "वाड्यात", "राज्य", "राज्यांत", "देश", "देशांत",
        "प्रदेश", "प्रदेशांत", "न्हंय", "न्हंयेत", "न्हंयेच्या", "तळें", "तळ्यांत",
        "दोंगर", "डोंगर", "पर्वत", "घाट", "घाटांत", "समुद्र", "सागरात", "किनारो", "बेट"
    },
    "assamese": {
        "জিলা", "জিলাৰ", "নগৰ", "নগৰত", "মহানগৰ", "মহানগৰত", "চহৰ", "চহৰত", "গাঁও",
        "গাঁৱত", "ৰাজ্য", "ৰাজ্যত", "দেশ", "দেশত", "প্ৰদেশ", "নদী", "নদীত", "নৈ", "নৈত",
        "পাহাৰ", "পাহাৰত", "পৰ্বত", "উপত্যকা", "দ্বীপ", "বিল", "জান", "ঘাট", "সাগৰ",
        "মহাসাগৰ", "ৰাষ্ট্ৰীয়", "উদ্যান", "অভয়াৰণ্য"
    },
    "sinhala": {
        "දිස්ත්‍රික්කය", "දිස්ත්‍රික්කයේ", "නගරය", "නගරයේ", "පළාත", "පළාතේ", "ගම", "ගමේ",
        "කොට්ඨාසය", "රාජ්‍යය", "රට", "ගංගාව", "ගංගාවේ", "ගඟ", "ඔය", "මුහුද", "මුහුදේ",
        "සාගරය", "සාගරයේ", "බොක්ක", "කන්ද", "කඳුකරය", "තැන්න", "දූපත", "වරාය", "වනෝද්‍යානය"
    },
}

# Geographic Modifiers that precede toponymic heads (North, South, Upper, Lower, etc.)
GEO_PREFIX_MODIFIERS: dict[str, set[str]] = {
    "konkani": {
        "उत्तर", "दक्षिण", "उदेंत", "अस्तंत", "व्हड", "धाकलो", "नवो", "पोरनो", "मध्य"
    },
    "assamese": {
        "উত্তৰ", "দক্ষিণ", "পূব", "পশ্চিম", "উজনি", "নামনি", "বৃহৎ", "নতুন", "পুৰণি", "মধ্য"
    },
    "sinhala": {
        "උතුරු", "දකුණු", "නැගෙනහිර", "බස්නාහිර", "මධ්‍යම", "වයඹ", "උතුරු-මැද", "මහා", "පැරණි", "නව"
    },
}


class DedicatedLOCRuleTagger:
    """
    High-precision morphological and syntactic rule tagger for geographic entities.
    """

    def __init__(self, lang: str, gazetteer_dir: Path | str | None = None) -> None:
        self.lang = lang.lower().strip()
        self.suffix_bases = TOPONYMIC_SUFFIX_BASES.get(self.lang, ())
        self.inflections = LOC_CASE_INFLECTIONS.get(self.lang, ())
        self.geo_heads = GEO_HEAD_KEYWORDS.get(self.lang, set())
        self.geo_prefixes = GEO_PREFIX_MODIFIERS.get(self.lang, set())

        # Load known place stems from gazetteer (filter out common adjectives/stopwords)
        COMMON_STOPWORDS = {
            "नवी", "नवो", "पोरनी", "पोरनो", "व्हड", "साल", "नदी", "নদী", "নতুন", "পুৰণি", "ගඟ", "නව", "මහා"
        }
        self.known_stems: set[str] = set()
        gaz_dir = Path(gazetteer_dir) if gazetteer_dir else PROJECT_ROOT / "data" / "gazetteers"
        places_file = gaz_dir / self.lang / "places.txt"
        if places_file.is_file():
            with open(places_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    entry = line.strip()
                    if entry and not entry.startswith("#"):
                        parts = entry.split()
                        if len(parts) == 1 and parts[0] not in COMMON_STOPWORDS and len(parts[0]) >= 3:
                            self.known_stems.add(parts[0])

        # Language-specific stem alternations for oblique/inflected cases
        if self.lang == "konkani":
            # Goa -> Goya, Goy -> Goya, Panaji -> Panaje
            self.known_stems.update(["पणजी", "गोया", "गोंया", "गोय", "गोंय", "पणजे", "म्हापशा", "फोंड्या", "मडगांव", "वास्को"])
        elif self.lang == "assamese":
            self.known_stems.update(["অসম", "গুৱাহাটী", "যোৰহাট", "তেজপুৰ", "শিলচৰ", "ব্ৰহ্মপুত্ৰ"])
        elif self.lang == "sinhala":
            self.known_stems.update(["කොළඹ", "නුවර", "ගාල්ල", "මාතර", "යාපනය", "මහනුවර"])

    def _matches_toponym_morpheme(self, token: str) -> bool:
        """
        Check if a token ends in a toponymic suffix base, either bare or with case inflection.
        """
        if token in PUNCTUATION_TOKENS or len(token) < 5:
            return False

        # 1. Bare suffix match (e.g. पणजी, मडगांव, গুৱাহাটী, කොළඹ)
        for base in self.suffix_bases:
            if token.endswith(base) and len(token) >= len(base) + 3:
                return True

        # 2. Inflected suffix match (e.g. गांव + ांत -> गांवांत, পুৰ + ত -> পুৰত, නුවර + ට -> නුවරට)
        for base in self.suffix_bases:
            for infl in self.inflections:
                comb = base + infl
                if token.endswith(comb) and len(token) >= len(comb) + 3:
                    return True

        return False

    def _matches_inflected_known_place(self, token: str) -> bool:
        """Check if token is an inflected form of a known base place stem."""
        if token in PUNCTUATION_TOKENS or len(token) < 4:
            return False

        for stem in self.known_stems:
            if len(stem) < 3:
                continue
            if token.startswith(stem) and len(token) > len(stem):
                rem = token[len(stem):]
                if rem in self.inflections:
                    return True
        return False

    def tag_sentence(self, tokens: list[str]) -> list[str]:
        """Apply morphological toponym rules and return valid BIO tags."""
        n = len(tokens)
        labels = ["O"] * n

        # Pass 1: Multi-token compound geographic structures
        # e.g. [उत्तर] [गोवा], [উজনি] [অসম], [දකුණු] [පළාත]
        for i in range(n):
            tok = tokens[i]
            if tok in PUNCTUATION_TOKENS:
                continue

            # Prefix modifier + Known Place: [North] + [Goa]
            if tok in self.geo_prefixes and i + 1 < n and tokens[i + 1] not in PUNCTUATION_TOKENS:
                next_tok = tokens[i + 1]
                if next_tok in self.known_stems or self._matches_toponym_morpheme(next_tok):
                    labels[i] = "B-LOC"
                    labels[i + 1] = "I-LOC"
                    continue

            # Toponym stem + Geographic Head: [Mandovi] + [River]
            if tok in self.geo_heads and i > 0 and tokens[i - 1] not in PUNCTUATION_TOKENS:
                prev_tok = tokens[i - 1]
                if prev_tok in self.known_stems or self._matches_toponym_morpheme(prev_tok):
                    if labels[i - 1] == "O":
                        labels[i - 1] = "B-LOC"
                        labels[i] = "I-LOC"
                        continue
                    elif labels[i - 1].startswith("B-LOC") or labels[i - 1].startswith("I-LOC"):
                        labels[i] = "I-LOC"
                        continue

        # Pass 2: Single-token morphological toponym and inflected place pattern matching
        for i in range(n):
            if labels[i] != "O" or tokens[i] in PUNCTUATION_TOKENS:
                continue
            tok = tokens[i]

            if self._matches_inflected_known_place(tok):
                labels[i] = "B-LOC"
            elif self._matches_toponym_morpheme(tok):
                labels[i] = "B-LOC"

        # Pass 3: Invariant cleanup (Punctuation always O, enforce BIO integrity)
        for i, t in enumerate(tokens):
            if t in PUNCTUATION_TOKENS:
                labels[i] = "O"

        for i in range(n):
            if labels[i].startswith("I-"):
                curr_type = labels[i][2:]
                if i == 0 or labels[i - 1] not in (f"B-{curr_type}", f"I-{curr_type}"):
                    labels[i] = f"B-{curr_type}"

        return labels
