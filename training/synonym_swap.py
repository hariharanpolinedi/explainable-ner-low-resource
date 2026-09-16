"""
Rule-Derived and Curated Synonym-Swap Data Augmentation Pipeline (Phase 11).

Replaces non-entity ('O' tagged) lexical tokens with in-language synonyms derived
from hand-curated vocabulary equivalence sets built from Phase 1/2 corpus statistics.
Guarantees entity spans and boundaries are 100% untouched and preserves grammatical structure.
Zero pretrained models or embeddings are used.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Hand-built / corpus-derived synonym equivalence clusters for non-entity tokens
CURATED_THESAURUS: dict[str, dict[str, list[str]]] = {
    "konkani": {
        "बरें": ["उत्तम", "सुंदर"],
        "उत्तम": ["बरें", "श्रेषठ"],
        "व्हड": ["महान", "विशाल"],
        "महान": ["व्हड", "विशाल"],
        "ल्हान": ["बारीक", "ल्हानशें"],
        "बारीक": ["ल्हान", "ल्हानशें"],
        "नवो": ["नव्या", "ताजो"],
        "नव्या": ["नवो", "ताजो"],
        "चड": ["खूप", "जास्त"],
        "खूप": ["चड", "पुष्कळ"],
        "दीस": ["वार", "दिवस"],
        "वर्स": ["साल", "संवत्सर"],
        "काळ": ["वेळ", "समयो"],
        "वेळ": ["काळ", "समयो"],
        "काम": ["वावर", "कार्य"],
        "वावर": ["काम", "उद्योग"],
        "घर": ["निवास", "सदन"],
        "गाव": ["ग्राम", "शार"],
        "मार्ग": ["रस्तो", "वाटो"],
        "रस्तो": ["मार्ग", "सडक"],
        "खास": ["विशेश", "खाशेलो"],
        "विशेश": ["खास", "महत्वाचो"],
        "जाग": ["स्थळ", "सुवात"],
        "सुवात": ["जाग", "ठिकाण"],
    },
    "assamese": {
        "ডাঙৰ": ["বৃহৎ", "বিশাল"],
        "বৃহৎ": ["ডাঙৰ", "বিশাল"],
        "ভাল": ["উত্তম", "সুন্দৰ"],
        "উত্তম": ["ভাল", "শ্ৰেষ্ঠ"],
        "নতুন": ["নব্য", "আধুনিক"],
        "নব্য": ["নতুন", "আধুনিক"],
        "বেছি": ["অধিক", "প্ৰচুৰ"],
        "অধিক": ["বেছি", "প্ৰচুৰ"],
        "সৰু": ["ক্ষুদ্ৰ", "কমলীয়া"],
        "ক্ষুদ্ৰ": ["সৰু", "কমলীয়া"],
        "পুৰণি": ["প্ৰাচীন", "ঐতিহাসিক"],
        "প্ৰাচীন": ["পুৰণি", "ঐতিহাসিক"],
        "প্ৰধান": ["মুখ্য", "প্ৰথম"],
        "মুখ্য": ["প্ৰধান", "শ্ৰেষ্ঠ"],
        "দিন": ["দিৱস", "বাৰ"],
        "দিৱস": ["দিন", "বাৰ"],
        "বছৰ": ["বৎসৰ", "সন"],
        "সময়": ["কাল", "বেলা"],
        "কাম": ["কাৰ্য", "কৰ্ম"],
        "কাৰ্য": ["কাম", "অনুষ্ঠান"],
        "ঘৰ": ["গৃহ", "নিবাস"],
        "গৃহ": ["ঘৰ", "আৱাস"],
        "বাট": ["পথ", "ৰাস্তা"],
        "পথ": ["বাট", "সড়ক"],
        "গাওঁ": ["গ্ৰাম", "জনপদ"],
    },
    "sinhala": {
        "විශාල": ["මහත්", "පුළුල්"],
        "මහත්": ["විශාල", "ප්‍රමුඛ"],
        "හොඳ": ["උසස්", "යහපත්"],
        "උසස්": ["හොඳ", "විශිෂ්ට"],
        "නව": ["අලුත්", "නූතන"],
        "අලුත්": ["නව", "නූතන"],
        "වැඩි": ["අධික", "බොහෝ"],
        "අධික": ["වැඩි", "බොහෝ"],
        "කුඩා": ["පොඩි", "සුළු"],
        "පොඩි": ["කුඩා", "සුළු"],
        "පැරණි": ["පුරාණ", "ඓතිහාසික"],
        "පුරාණ": ["පැරණි", "ඓතිහාසික"],
        "ප්‍රධාන": ["මූලික", "ප්‍රමුඛ"],
        "මූලික": ["ප්‍රධාන", "මුල්"],
        "දින": ["දිනය", "වාරය"],
        "වසර": ["අවුරුද්ද", "සංවත්සරය"],
        "අවුරුද්ද": ["වසර", "සංවත්සරය"],
        "කාලය": ["වේලාව", "සමය"],
        "වේලාව": ["කාලය", "මොහොත"],
        "වැඩ": ["කාර්යය", "ක්‍රියාව"],
        "කාර්යය": ["වැඩ", "මෙහෙවර"],
        "නිවස": ["ගෙදර", "වාසස්ථානය"],
        "ගෙදර": ["නිවස", "වාසස්ථානය"],
        "මඟ": ["පාර", "මාර්ගය"],
        "පාර": ["මඟ", "වීදිය"],
        "ගම": ["ගම්මානය", "ප්‍රදේශය"],
    },
}


class SynonymSwapAugmenter:
    """
    Performs synonym swap on non-entity ('O') tokens.
    Guarantees entity tokens and boundary tags remain 100% unaltered.
    """

    def __init__(self, lang: str, seed: int = 42, swap_prob: float = 0.25) -> None:
        self.lang = lang.lower()
        self.swap_prob = swap_prob
        self.rng = random.Random(seed)
        self.thesaurus = CURATED_THESAURUS.get(self.lang, {})

    def augment_sentence(
        self,
        tokens: list[str],
        tags: list[str],
        max_swaps: int = 2,
    ) -> tuple[list[str], list[str], bool]:
        """
        Swap up to max_swaps eligible non-entity tokens with a synonym.

        Returns:
            (augmented_tokens, tags, was_augmented)
        """
        if not self.thesaurus or len(tokens) != len(tags):
            return list(tokens), list(tags), False

        # Identify candidate indices: must have tag 'O' and exist in thesaurus
        eligible_indices = [
            i for i, (tok, tag) in enumerate(zip(tokens, tags))
            if tag == "O" and tok in self.thesaurus and self.thesaurus[tok]
        ]

        if not eligible_indices:
            return list(tokens), list(tags), False

        # Sample up to max_swaps indices
        num_swaps = min(len(eligible_indices), max_swaps)
        chosen_indices = set(self.rng.sample(eligible_indices, num_swaps))

        new_tokens = list(tokens)
        augmented = False

        for idx in chosen_indices:
            orig_word = tokens[idx]
            synonyms = self.thesaurus[orig_word]
            if synonyms:
                replacement = self.rng.choice(synonyms)
                new_tokens[idx] = replacement
                augmented = True

        return new_tokens, list(tags), augmented

    def generate_augmented_dataset(
        self,
        sentences: list[tuple[list[str], list[str]]],
        multiplier: int = 1,
    ) -> list[tuple[list[str], list[str]]]:
        """
        Generate synonym-swapped versions of training sentences.
        """
        augmented_set: list[tuple[list[str], list[str]]] = []
        for toks, tags in sentences:
            for _ in range(multiplier):
                aug_toks, aug_tags, ok = self.augment_sentence(toks, tags)
                if ok:
                    augmented_set.append((aug_toks, aug_tags))
        return augmented_set


if __name__ == "__main__":
    test_toks = ["डॉ.", "नागेश", "करमली", "व्हड", "सुवात", "गेले", "."]
    test_tags = ["B-PER", "I-PER", "I-PER", "O", "O", "O", "O"]
    swapper = SynonymSwapAugmenter("konkani", seed=42)
    new_toks, new_tags, ok = swapper.augment_sentence(test_toks, test_tags)
    print("Original:", test_toks)
    print("Swapped (ok={}):".format(ok), new_toks)
