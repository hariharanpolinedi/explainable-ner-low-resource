"""
Gazetteer-based weak supervision labeler for low-resource languages.

Performs exact substring/token-sequence matching of sentences against
Phase 1 gazetteer entries (PER, LOC, ORG) and cultural/statutory MISC lexicons.
Includes configurable stoplist filtering to prevent over-matching of ambiguous common words.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import BaseConfig, load_config

PUNCTUATION_TOKENS = {
    ".", ",", "!", "?", "।", "॥", ";", ":", "-", "—", "(", ")",
    "[", "]", "{", "}", "\"", "'", "«", "»", "෴", "%", "/", "@"
}

CULTURAL_MISC: dict[str, list[str]] = {
    "konkani": [
        "शिगमो", "चवथ", "दिवाळी", "नाताल", "साहित्य अकादेमी पुरस्कार", "अकादेमी पुरस्कार",
        "गोंय मुक्ती दीस", "ज्ञानपीठ पुरस्कार", "पद्मश्री", "पद्मभूषण", "संविधान", "रामायण", "महाभारत"
    ],
    "assamese": [
        "ৰঙালী বিহু", "ভোগালী বিহু", "কঙালী বিহু", "বিহু", "পদ্মশ্ৰী", "পদ্মভূষণ",
        "জ্ঞানপীঠ বঁটা", "অসম চুক্তি", "বুঢ়ী আইৰ সাধু", "সংবিধান", "ৰামায়ণ", "মহাভাৰত"
    ],
    "sinhala": [
        "මහාවංශය", "සිංහල අලුත් අවුරුද්ද", "අවුරුද්ද", "දළදා පෙරහැර", "පෙරහැර", "වෙසක්",
        "පොසොන්", "රණවිරු සම්මානය", "ආණ්ඩුක්‍රම ව්‍යවස්ථාව", "රාමායණය", "මහාභාරතය", "ත්‍රිපිටකය"
    ],
}


class GazetteerMatcher:
    """Exact-match dictionary labeler with longest-match-first priority."""

    def __init__(
        self,
        lang: str,
        gazetteer_dir: Path | str,
        stoplist: list[str] | set[str] | None = None,
    ) -> None:
        self.lang = lang.lower()
        self.gazetteer_dir = Path(gazetteer_dir)
        self.stoplist = set(stoplist or [])
        self.entities: dict[str, list[list[str]]] = {"ORG": [], "MISC": [], "PER": [], "LOC": []}
        self._load_gazetteers()

    def _load_gazetteers(self) -> None:
        lang_dir = self.gazetteer_dir / self.lang
        cat_map = {"orgs": "ORG", "persons": "PER", "places": "LOC"}

        for cat_file, etype in cat_map.items():
            fpath = lang_dir / f"{cat_file}.txt"
            if fpath.is_file():
                with open(fpath, "r", encoding="utf-8") as fp:
                    for line in fp:
                        entry = line.strip()
                        if not entry or entry in self.stoplist or len(entry) < 2:
                            continue
                        tokens = entry.split()
                        # Avoid single-token stoplist matches
                        if len(tokens) == 1 and tokens[0] in self.stoplist:
                            continue
                        self.entities[etype].append(tokens)

        # Add cultural MISC lexicons
        for item in CULTURAL_MISC.get(self.lang, []):
            if item not in self.stoplist:
                tokens = item.split()
                self.entities["MISC"].append(tokens)

        # Sort each list by descending token length so longest spans match first
        for etype in self.entities:
            self.entities[etype].sort(key=len, reverse=True)

    def tag_sentence(self, tokens: list[str]) -> list[str]:
        """
        Label a token sequence using exact dictionary lookup.
        Returns a list of BIO tags of identical length.
        """
        n = len(tokens)
        labels = ["O"] * n
        matched = [False] * n

        # Match in priority order: ORG, MISC, PER, LOC
        for etype in ["ORG", "MISC", "PER", "LOC"]:
            for span_tokens in self.entities[etype]:
                span_len = len(span_tokens)
                if span_len > n:
                    continue

                for i in range(n - span_len + 1):
                    if any(matched[i + k] for k in range(span_len)):
                        continue

                    # Boundary match check
                    if tokens[i : i + span_len] == span_tokens:
                        # Punctuation boundary invariant
                        if tokens[i] in PUNCTUATION_TOKENS or tokens[i + span_len - 1] in PUNCTUATION_TOKENS:
                            continue

                        labels[i] = f"B-{etype}"
                        matched[i] = True
                        for k in range(1, span_len):
                            labels[i + k] = f"I-{etype}"
                            matched[i + k] = True

        # Invariant: Punctuation is always O
        for i, tok in enumerate(tokens):
            if tok in PUNCTUATION_TOKENS:
                labels[i] = "O"

        # BIO integrity check
        for i in range(n):
            if labels[i].startswith("I-"):
                curr_type = labels[i][2:]
                if i == 0 or labels[i - 1] not in (f"B-{curr_type}", f"I-{curr_type}"):
                    labels[i] = f"B-{curr_type}"

        return labels
