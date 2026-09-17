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
        org_oversample_ratio: float = 3.0,
    ) -> tuple[list[str], list[str], bool]:
        """
        Substitute entity spans in a sentence with matching gazetteer entries,
        optionally oversampling ORG spans.
        """
        spans = sorted(list(extract_entities(tags)), key=lambda s: s[1])
        if not spans:
            return list(tokens), list(tags), False

        # Weight selection towards ORG entities
        weights = [org_oversample_ratio if s[0] == "ORG" else 1.0 for s in spans]
        tot_w = sum(weights)
        norm_weights = [w / tot_w for w in weights]

        k = min(len(spans), max_substitutions)
        # Sample without replacement using weighted distribution
        chosen_indices = set()
        for _ in range(k):
            # Pick an unselected span
            unselected = [idx for idx in range(len(spans)) if idx not in chosen_indices]
            if not unselected:
                break
            sub_w = [weights[idx] for idx in unselected]
            sub_tot = sum(sub_w)
            sub_probs = [w / sub_tot for w in sub_w]
            chosen_idx = self.rng.choices(unselected, weights=sub_probs, k=1)[0]
            chosen_indices.add(chosen_idx)

        chosen_spans = [spans[idx] for idx in sorted(chosen_indices)]

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
                new_tokens.extend(orig_entity_tokens)
                new_tags.extend(tags[start:end])

            cursor = end

        # Append remaining tokens
        new_tokens.extend(tokens[cursor:])
        new_tags.extend(tags[cursor:])

        return new_tokens, new_tags, augmented

    def generate_multi_entity_list_sentence(
        self,
        target_entity_types: Sequence[str] = ("ORG", "ORG", "ORG"),
    ) -> tuple[list[str], list[str]]:
        """
        Synthesize longer multi-clause sentences containing entities separated by commas
        and conjunctions to train the model on middle-entity boundary recognition.
        """
        # Templates per language: (prefix_tokens, comma_token, conj_token, suffix_tokens)
        templates = {
            "konkani": [
                ([], ",", "आनी", ["हाणीं", "संयुक्त", "बसका", "आयोजीत", "केली", "."]),
                (["गोंय", "सरकारान"], ",", "आनी", ["हांकां", "विशेष", "अनुदान", "जाहीर", "केलें", "."]),
                ([], ",", "आनी", ["हे", "गोंयच्या", "सांस्कृतिक", "आनी", "शैक्षणिक", "मळार", "वावुरतात", "."]),
                (["गोंयच्या", "प्रशासनान"], ",", "आनी", ["हांच्या", "प्रतिनिधींक", "सचिवालयांत", "आपयले", "."]),
            ],
            "assamese": [
                ([], ",", "আৰু", ["যৌথভাৱে", "এক", "গুৰুত্বপূৰ্ণ", "বৈঠকত", "মিলিত", "হয়", "।"]),
                (["অসম", "চৰকাৰে"], ",", "আৰু", ["ক", "নতুন", "উন্নয়নমূলক", "আঁচনিৰ", "দায়িত্ব", "অৰ্পণ", "কৰিছে", "।"]),
                ([], ",", "আৰু", ["অসমৰ", "ঐতিহ্যমণ্ডিত", "আৰু", "সন্মানীয়", "অনুষ্ঠান", "হিচাপে", "পৰিচিত", "।"]),
                (["গুৱাহাটীত"], ",", "আৰু", ["ৰ", "উদ্যোগত", "এখন", "ৰাজ্যিক", "পৰ্যায়ৰ", "সভা", "অনুষ্ঠিত", "হয়", "।"]),
            ],
            "sinhala": [
                ([], ",", "සහ", ["එක්ව", "විශේෂ", "සාකච්ඡාවක්", "පැවැත්වීමට", "තීරණය", "කළේය", "."]),
                (["ශ්‍රී", "ලංකා", "රජය"], ",", "සහ", ["සඳහා", "නව", "ප්‍රතිපත්ති", "ක්‍රියාත්මක", "කිරීමට", "සැලසුම්", "කරයි", "."]),
                ([], ",", "සහ", ["රටේ", "ප්‍රමුඛ", "ආයතන", "ලෙස", "කටයුතු", "කරයි", "."]),
                (["කොළඹදී"], ",", "සහ", ["නියෝජිතයන්", "හමුවී", "සාකච්ඡා", "කළහ", "."]),
            ],
        }

        lang_temps = templates.get(self.lang, templates["konkani"])
        prefix, comma_str, conj_str, suffix = self.rng.choice(lang_temps)

        # Pick entities
        ents: list[tuple[str, list[str]]] = []
        for etype in target_entity_types:
            pool = self.gazetteers.get(etype, [])
            if pool:
                chosen = self.rng.choice(pool)
                ents.append((etype, chosen))
            else:
                ents.append((etype, ["X"]))

        tokens: list[str] = list(prefix)
        tags: list[str] = ["O"] * len(prefix)

        # E1
        tokens.extend(ents[0][1])
        tags.append(f"B-{ents[0][0]}")
        tags.extend([f"I-{ents[0][0]}"] * (len(ents[0][1]) - 1))

        # comma
        tokens.append(comma_str)
        tags.append("O")

        # E2 (Middle entity!)
        tokens.extend(ents[1][1])
        tags.append(f"B-{ents[1][0]}")
        tags.extend([f"I-{ents[1][0]}"] * (len(ents[1][1]) - 1))

        # conjunction
        tokens.append(conj_str)
        tags.append("O")

        # E3
        if len(ents) >= 3:
            tokens.extend(ents[2][1])
            tags.append(f"B-{ents[2][0]}")
            tags.extend([f"I-{ents[2][0]}"] * (len(ents[2][1]) - 1))

        # suffix
        tokens.extend(suffix)
        tags.extend(["O"] * len(suffix))

        return tokens, tags

    def generate_augmented_dataset(
        self,
        sentences: list[tuple[list[str], list[str]]],
        multiplier: int = 1,
        include_multi_entity_lists: bool = True,
        num_multi_entity_lists: int = 200,
        org_oversample_ratio: float = 3.0,
    ) -> list[tuple[list[str], list[str]]]:
        """
        Generate synthetic entity-substituted sentences including targeted multi-entity lists.
        """
        augmented_set: list[tuple[list[str], list[str]]] = []

        # 1. Standard Substitution with ORG oversampling
        for toks, tags in sentences:
            has_entity = any(t != "O" for t in tags)
            if not has_entity:
                continue

            for _ in range(multiplier):
                aug_toks, aug_tags, success = self.augment_sentence(
                    toks, tags, org_oversample_ratio=org_oversample_ratio
                )
                if success:
                    augmented_set.append((aug_toks, aug_tags))

        # 2. Multi-Entity List Synthesis (Targeting comma/conjunction lists)
        if include_multi_entity_lists:
            # Generate combinations: (ORG, ORG, ORG), (PER, ORG, LOC), (ORG, LOC, ORG), etc.
            combinations = [
                ("ORG", "ORG", "ORG"),
                ("ORG", "ORG", "LOC"),
                ("LOC", "ORG", "ORG"),
                ("PER", "ORG", "LOC"),
                ("ORG", "PER", "ORG"),
            ]
            for _ in range(num_multi_entity_lists):
                combo = self.rng.choice(combinations)
                l_toks, l_tags = self.generate_multi_entity_list_sentence(target_entity_types=combo)
                augmented_set.append((l_toks, l_tags))

        return augmented_set



if __name__ == "__main__":
    test_toks = ["डॉ.", "नागेश", "करमली", "गोंयांत", "गेले", "."]
    test_tags = ["B-PER", "I-PER", "I-PER", "B-LOC", "O", "O"]
    augmenter = EntitySubstitutionAugmenter("konkani", seed=42)
    aug_toks, aug_tags, ok = augmenter.augment_sentence(test_toks, test_tags)
    print("Original:", test_toks, test_tags)
    print("Augmented (ok={}):".format(ok), aug_toks, aug_tags)
