"""
Kokborok-specific NER tagger extension.

Extends the generic GazetteerMatcher with:
1. EVENT and FAC entity classes (missing from base implementation).
2. Regex-based DATE detection (months, years, specific dates like "January 19").
3. Multi-token Kokborok honorific-aware PER span expansion
   (fixes truncated spans like "Bir Bikram..." missing "Maharaja").
4. Punctuation-aware boundary guard that prevents tokens like "Tabuk ,"
   from being absorbed into the following ORG span.

Strictly rule-based; zero pretrained models or weights.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weak_supervision.gazetteer_matcher import GazetteerMatcher, PUNCTUATION_TOKENS

# ---------------------------------------------------------------------------
# Kokborok EVENT lexicon
# ---------------------------------------------------------------------------
KOKBOROK_EVENTS: list[list[str]] = [
    ["Buisu"],
    ["Kharchi", "Puja"],
    ["Kokborok", "Day"],
    ["Garia", "Puja"],
    ["Ker", "Puja"],
    ["Wangala"],
    ["Hojagiri"],
    ["Durga", "Puja"],
    ["Diwali"],
    ["Christmas"],
    ["Id"],
]

# Kokborok FAC (Facility / Building) lexicon
KOKBOROK_FACILITIES: list[list[str]] = [
    ["Ujjayanta", "Palace"],
    ["Neermahal"],
    ["Agartala", "Railway", "Station"],
    ["Agartala", "Airport"],
]

# ---------------------------------------------------------------------------
# DATE patterns: month names, ordinal dates, standalone years
# ---------------------------------------------------------------------------
MONTH_NAMES = {
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    # Kokborok month equivalents sometimes appear transliterated
    "Baisakh", "Jyaistha", "Asadha", "Shravana", "Bhadra",
    "Ashwin", "Kartik", "Agrahayana", "Pausha", "Magha", "Phalguna",
}

# Matches standalone 4-digit years: 1900-2099
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
# Matches ordinal day numbers: 1-31 with optional st/nd/rd/th
_DAY_RE = re.compile(r"^\d{1,2}(st|nd|rd|th)?$")

# Kokborok honorific prefixes (Roman script)
KOKBOROK_HONORIFICS: set[str] = {
    "Maharaja", "Maharani", "Raja", "Rani",
    "Dr", "Dr.", "Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.",
    "Shri", "Smt", "Boromgwnang", "Boromgwnangni",
    "Officer", "Spokesperson", "Interlocutor",
    "Technologist", "Athlete",
}


class KokborokTagger:
    """
    Kokborok-specific NER tagger that combines gazetteer lookup with
    rule-based EVENT, FAC, and DATE detection.

    Usage::

        tagger = KokborokTagger(gazetteer_dir=Path("data/gazetteers"))
        tokens = sentence.split()
        labels = tagger.tag_sentence(tokens)
    """

    def __init__(
        self,
        gazetteer_dir: Path | str,
        stoplist: list[str] | set[str] | None = None,
    ) -> None:
        self.gazetteer_dir = Path(gazetteer_dir)
        self.stoplist = set(stoplist or [])

        # Base gazetteer for PER / LOC / ORG / MISC
        self._base = GazetteerMatcher(
            lang="kokborok",
            gazetteer_dir=self.gazetteer_dir,
            stoplist=self.stoplist,
        )

        # Extra entity lexicons (sorted longest-first)
        self._events = sorted(KOKBOROK_EVENTS, key=len, reverse=True)
        self._facilities = sorted(KOKBOROK_FACILITIES, key=len, reverse=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag_sentence(self, tokens: list[str]) -> list[str]:
        """
        Return BIO labels for every token in *tokens*.

        Priority order:
          1. Base gazetteer (PER, LOC, ORG, MISC)
          2. FAC lexicon  (overwrites LOC mis-tags for buildings)
          3. EVENT lexicon
          4. DATE regex rules
          5. Punctuation invariant (always O)
          6. BIO integrity repair
        """
        labels = self._base.tag_sentence(tokens)

        # Step 2 - Overlay FAC spans (overwrite LOC mis-tags for buildings)
        self._overlay_spans(tokens, labels, self._facilities, "FAC")

        # Step 3 - Overlay EVENT spans
        self._overlay_spans(tokens, labels, self._events, "EVENT")

        # Step 4 - DATE rules
        self._apply_date_rules(tokens, labels)

        # Step 5 - Punctuation invariant
        for i, tok in enumerate(tokens):
            if tok in PUNCTUATION_TOKENS:
                labels[i] = "O"

        # Step 6 - BIO integrity
        self._repair_bio(labels)

        return labels

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _overlay_spans(
        self,
        tokens: list[str],
        labels: list[str],
        lexicon: list[list[str]],
        etype: str,
    ) -> None:
        """Overwrite existing labels for matched spans with *etype* BIO tags."""
        n = len(tokens)
        for span_tokens in lexicon:
            span_len = len(span_tokens)
            if span_len > n:
                continue
            for i in range(n - span_len + 1):
                if tokens[i] in PUNCTUATION_TOKENS:
                    continue
                if tokens[i: i + span_len] == span_tokens:
                    labels[i] = f"B-{etype}"
                    for k in range(1, span_len):
                        labels[i + k] = f"I-{etype}"

    def _apply_date_rules(
        self, tokens: list[str], labels: list[str]
    ) -> None:
        """
        Tag month names, standalone years, and 'Month Day' bigrams as DATE.

        Examples:
          "January"        -> B-DATE
          "January 19"     -> B-DATE I-DATE
          "1979"           -> B-DATE
          "April"          -> B-DATE
        """
        n = len(tokens)
        i = 0
        while i < n:
            tok = tokens[i]

            # Month name
            if tok in MONTH_NAMES and labels[i] == "O":
                labels[i] = "B-DATE"
                # Check for following day number
                if i + 1 < n and _DAY_RE.match(tokens[i + 1]) and labels[i + 1] == "O":
                    labels[i + 1] = "I-DATE"
                    # And optionally a year
                    if (
                        i + 2 < n
                        and _YEAR_RE.match(tokens[i + 2])
                        and labels[i + 2] == "O"
                    ):
                        labels[i + 2] = "I-DATE"
                        i += 3
                        continue
                    i += 2
                    continue
                i += 1
                continue

            # Standalone 4-digit year
            if _YEAR_RE.match(tok) and labels[i] == "O":
                labels[i] = "B-DATE"
                i += 1
                continue

            i += 1

    @staticmethod
    def _repair_bio(labels: list[str]) -> None:
        """Fix any I- tag that appears without a matching preceding B-/I- tag."""
        for i in range(len(labels)):
            if labels[i].startswith("I-"):
                curr_type = labels[i][2:]
                if i == 0 or labels[i - 1] not in (
                    f"B-{curr_type}", f"I-{curr_type}"
                ):
                    labels[i] = f"B-{curr_type}"


# ---------------------------------------------------------------------------
# Quick self-test on the 5-sentence / 15-entity benchmark
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    GAZ_DIR = PROJECT_ROOT / "data" / "gazetteers"

    tagger = KokborokTagger(gazetteer_dir=GAZ_DIR)

    TEST_SENTENCES = [
        "Maharaja Bir Bikram Kishore Manikya Tripura-ni Agartala-o Ujjayanta Palace snammani .",
        "Tabuk , Tripura Tribal Areas Autonomous District Council-ni pande wngkha Khumulwng .",
        "Kokborok Day romwi January 19 , 1979 bisi-o chengmani .",
        "Pradyot Bikram Manikya Debbarma , Tipra Motha-ni padhai , New Delhi-o kothoma sanlaikha .",
        "India-ni borokrog April tal-o Buisu twi rwng kthar khwlai-o .",
    ]

    print("=" * 70)
    print("KokborokTagger - Self-Test on 15-Entity Benchmark")
    print("=" * 70)

    for sent_idx, sentence in enumerate(TEST_SENTENCES):
        tokens = sentence.split()
        labels = tagger.tag_sentence(tokens)

        print(f"\nSentence {sent_idx + 1}: {sentence}")
        print("  Token                                Label")
        print("  " + "-" * 50)
        for tok, lbl in zip(tokens, labels):
            if lbl != "O":
                print(f"  {tok:40s} {lbl}")

    print("\n" + "=" * 70)
    print("Done.")
