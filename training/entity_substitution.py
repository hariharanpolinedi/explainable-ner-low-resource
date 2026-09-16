"""
Entity-Substitution Data Augmentation Pipeline (Phase 11).

Generates synthetic training sentences by replacing gold/silver entity spans
with gazetteer entries of the same entity category (PER, LOC, ORG, MISC).
Incorporates rule-derived agreement heuristics per language (honorifics,
gender/classifier agreement, structural length) and guarantees strict BIO tag integrity.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.seqeval_metrics import extract_entities
from preprocessing.tokenizer import ScriptTokenizer
from weak_supervision.gazetteer_matcher import CULTURAL_MISC


class EntitySubstitutionAugmenter:
    """
    Performs category-consistent entity span substitution with language-specific
    agreement preservation and rigorous BIO tag reassignment.
    """

    def __init__(
        self,
        lang: str,
        gazetteer_dir: Path | str = "data/gazetteers",
        seed: int = 42,
        check_agreement: bool = True,
    ) -> None:
        self.lang = lang.lower()
        self.gazetteer_dir = Path(gazetteer_dir)
        self.check_agreement = check_agreement
        self.rng = random.Random(seed)
        self.tokenizer = ScriptTokenizer(self.lang)

        # Category -> list of tokenized entries
        self.gazetteers: dict[str, list[list[str]]] = {
            "PER": [],
            "LOC": [],
            "ORG": [],
            "MISC": [],
        }
        self._load_gazetteers()

    def _load_gazetteers(self) -> None:
        """Load and tokenize all gazetteer entries per category."""
        lang_dir = self.gazetteer_dir / self.lang

        file_map = {
            "PER": "persons.txt",
            "LOC": "places.txt",
            "ORG": "orgs.txt",
        }

        for cat, fname in file_map.items():
            fpath = lang_dir / fname
            if fpath.is_file():
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = line.strip()
                        if entry:
                            toks = self.tokenizer.tokenize_words(entry)
                            if toks:
                                self.gazetteers[cat].append(toks)

        # Load MISC from cultural lexicons
        misc_entries = CULTURAL_MISC.get(self.lang, [])
        for entry in misc_entries:
            toks = self.tokenizer.tokenize_words(entry)
            if toks:
                self.gazetteers["MISC"].append(toks)

    def _classify_agreement(self, etype: str, tokens: list[str]) -> str:
        """
        Derive language-specific grammatical agreement signature:
        - Honorific / gender markers for PER
        - Suffix / enclitic patterns for LOC / ORG
        """
        if not self.check_agreement or not tokens:
            return "DEFAULT"

        text = " ".join(tokens)

        if self.lang == "konkani":
            # Check honorifics or gender markers
            if any(t in tokens for t in ["डॉ.", "प्रा.", "श्री.", "श्रीमती."]):
                return "HONORIFIC"
            if text.endswith("बाई") or text.endswith("देवी") or text.endswith("ी"):
                return "FEMININE"
            if text.endswith("राव") or text.endswith("बाबू") or text.endswith("ो"):
                return "MASCULINE"

        elif self.lang == "assamese":
            # Check classifiers or honorifics
            if any(t in tokens for t in ["ড°", "ডাঃ", "শ্ৰী", "শ্ৰীমতী", "প্ৰাঃ"]):
                return "HONORIFIC"
            if any(text.endswith(sfx) for sfx in ["জন", "জনা", "গৰাকী"]):
                return "HUMAN_CLASSIFIER"
            if any(text.endswith(sfx) for sfx in ["খন", "টো", "ডাল"]):
                return "INANIMATE_CLASSIFIER"

        elif self.lang == "sinhala":
            # Check honorifics or case markers
            if any(t in tokens for t in ["ගරු", "පූජ්‍ය", "මහාචාර්ය", "ඩොක්ටර්"]):
                return "HONORIFIC"
            if text.endswith("තුමා") or text.endswith("මයා"):
                return "HONORIFIC_MASC"
            if text.endswith("තුමිය") or text.endswith("මිය"):
                return "HONORIFIC_FEM"

        return "DEFAULT"

    def augment_sentence(
        self,
        tokens: list[str],
        tags: list[str],
        max_substitutions: int = 2,
    ) -> tuple[list[str], list[str], bool]:
        """
        Substitute entity spans in a sentence with matching gazetteer entries.

        Returns:
            (augmented_tokens, augmented_tags, was_augmented)
        """
        spans = sorted(list(extract_entities(tags)), key=lambda s: s[1])
        if not spans:
            return list(tokens), list(tags), False

        # Randomly choose up to max_substitutions spans to replace
        chosen_spans = self.rng.sample(spans, min(len(spans), max_substitutions))
        chosen_spans.sort(key=lambda s: s[1])

        new_tokens: list[str] = []
        new_tags: list[str] = []
        cursor = 0
        augmented = False

        for etype, start, end in chosen_spans:
            # Append preceding tokens
            new_tokens.extend(tokens[cursor:start])
            new_tags.extend(tags[cursor:start])

            orig_entity_tokens = tokens[start:end]
            pool = self.gazetteers.get(etype, [])

            if self.check_agreement:
                sig = self._classify_agreement(etype, orig_entity_tokens)
                matching_pool = [e for e in pool if self._classify_agreement(etype, e) == sig]
                candidate_pool = matching_pool if len(matching_pool) >= 2 else pool
            else:
                candidate_pool = pool

            # Exclude identical replacement
            candidates = [c for c in candidate_pool if c != orig_entity_tokens]
            if candidates:
                replacement = self.rng.choice(candidates)
                new_tokens.extend(replacement)
                new_tags.append(f"B-{etype}")
                for _ in range(1, len(replacement)):
                    new_tags.append(f"I-{etype}")
                augmented = True
            else:
                # Keep original if no candidate available
                new_tokens.extend(orig_entity_tokens)
                new_tags.extend(tags[start:end])

            cursor = end

        # Append remaining tokens
        new_tokens.extend(tokens[cursor:])
        new_tags.extend(tags[cursor:])

        return new_tokens, new_tags, augmented

    def generate_augmented_dataset(
        self,
        sentences: list[tuple[list[str], list[str]]],
        multiplier: int = 1,
    ) -> list[tuple[list[str], list[str]]]:
        """
        Generate synthetic entity-substituted sentences.
        """
        augmented_set: list[tuple[list[str], list[str]]] = []
        for toks, tags in sentences:
            has_entity = any(t != "O" for t in tags)
            if not has_entity:
                continue

            for _ in range(multiplier):
                aug_toks, aug_tags, success = self.augment_sentence(toks, tags)
                if success:
                    augmented_set.append((aug_toks, aug_tags))

        return augmented_set


if __name__ == "__main__":
    test_toks = ["डॉ.", "नागेश", "करमली", "गोंयांत", "गेले", "."]
    test_tags = ["B-PER", "I-PER", "I-PER", "B-LOC", "O", "O"]
    augmenter = EntitySubstitutionAugmenter("konkani", seed=42)
    aug_toks, aug_tags, ok = augmenter.augment_sentence(test_toks, test_tags)
    print("Original:", test_toks, test_tags)
    print("Augmented (ok={}):".format(ok), aug_toks, aug_tags)
