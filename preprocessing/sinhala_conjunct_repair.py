"""
Deterministic Rule-Based Sinhala Conjunct Repair Normalizer.

Detects and repairs decomposed / ASCII-like conjunct sequences in Sinhala text
where Zero-Width Joiner (ZWJ, U+200D) was omitted during typing:
1. Rakaransaya (ර-කාරාංශය):
   Pattern: [Consonant != Ra] + Virama (\u0DCA) + Ra (\u0DBB) without ZWJ
   Repair: Inserts ZWJ -> Consonant + Virama + ZWJ + Ra (e.g. ප්ර -> ප්‍ර, ශ්ර -> ශ්‍ර, ග්ර -> ග්‍ර)
2. Yansaya (ය-අංශය):
   Pattern: [Consonant != Ra] + Virama (\u0DCA) + Ya (\u0DBA) without ZWJ
   Repair: Inserts ZWJ -> Consonant + Virama + ZWJ + Ya (e.g. ද්ය -> ද්‍ය, ත්ය -> ත්‍ය)
3. De-conjugation / Stripping (for training-time robustness augmentation):
   Reverses the canonical formation to generate realistic noisy/decomposed inputs.

Zero pretrained models, zero external tokenizers. Pure Unicode rule logic.
Strictly script-gated: only operates on Sinhala code points (\u0D80-\u0DFF).
"""

from __future__ import annotations

import re
import sys
from typing import Sequence

# Sinhala Unicode block: U+0D80 to U+0DFF
SINHALA_CHAR_REGEX = re.compile(r"[\u0D80-\u0DFF]")

# Sinhala consonants: U+0D9A to U+0DC6 (excluding independent vowels U+0D85-U+0D96)
# Consonants other than Ra (\u0DBB):
SINHALA_CONSONANTS_NO_RA = r"[\u0D9A-\u0DBA\u0DBC-\u0DC6]"

VIRAMA = "\u0DCA"
ZWJ = "\u200D"
RA = "\u0DBB"
YA = "\u0DBA"

# Explicit Conjunct Formation Rule Table
SINHALA_CONJUNCT_RULES = [
    {
        "name": "Rakaransaya",
        "description": "Consonant + Virama + Ra -> Consonant + Virama + ZWJ + Ra",
        "pattern": re.compile(f"({SINHALA_CONSONANTS_NO_RA}){VIRAMA}(?!{ZWJ}){RA}"),
        "replacement": lambda m: f"{m.group(1)}{VIRAMA}{ZWJ}{RA}",
    },
    {
        "name": "Yansaya",
        "description": "Consonant (except Ra) + Virama + Ya -> Consonant + Virama + ZWJ + Ya",
        "pattern": re.compile(f"({SINHALA_CONSONANTS_NO_RA}){VIRAMA}(?!{ZWJ}){YA}"),
        "replacement": lambda m: f"{m.group(1)}{VIRAMA}{ZWJ}{YA}",
    },
]

# Reverse De-conjugation patterns (for robustness augmentation)
SINHALA_DECOMPOSE_RULES = [
    {
        "name": "Decompose_Rakaransaya",
        "pattern": re.compile(f"({SINHALA_CONSONANTS_NO_RA}){VIRAMA}{ZWJ}{RA}"),
        "replacement": lambda m: f"{m.group(1)}{VIRAMA}{RA}",
    },
    {
        "name": "Decompose_Yansaya",
        "pattern": re.compile(f"({SINHALA_CONSONANTS_NO_RA}){VIRAMA}{ZWJ}{YA}"),
        "replacement": lambda m: f"{m.group(1)}{VIRAMA}{YA}",
    },
]


def is_sinhala_text(text: str) -> bool:
    """Check if the text contains any Sinhala script characters."""
    return bool(SINHALA_CHAR_REGEX.search(text))


def repair_sinhala_conjuncts(text: str) -> str:
    """
    Deterministically repair decomposed Sinhala conjuncts by inserting ZWJ.
    Guaranteed no-op on non-Sinhala text (Konkani, Assamese, English).
    """
    if not text or not is_sinhala_text(text):
        return text

    repaired = text
    for rule in SINHALA_CONJUNCT_RULES:
        repaired = rule["pattern"].sub(rule["replacement"], repaired)

    return repaired


def decompose_sinhala_conjuncts(text: str) -> str:
    """
    Strip ZWJ from Sinhala conjuncts to simulate realistic un-normalized user input.
    Used exclusively for training-time robustness data augmentation.
    """
    if not text or not is_sinhala_text(text):
        return text

    decomposed = text
    for rule in SINHALA_DECOMPOSE_RULES:
        decomposed = rule["pattern"].sub(rule["replacement"], decomposed)

    return decomposed


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sample = "මහාචාර්ය අනුර සිල්වා සහ ආචාර්ය නිමාලි ප්රනාන්දු විසින් පේරාදෙණිය විශ්වවිද්යාලයේදී සංවර්ධනය කරන ලද පද්ධතිය ශ්රී ලංකාවේ ග්රාමීය ප්රදේශවල..."
    repaired = repair_sinhala_conjuncts(sample)
    print("Original:", sample[:60])
    print("Repaired:", repaired[:60])
    decomposed = decompose_sinhala_conjuncts(repaired)
    print("Decomposed:", decomposed[:60])
    assert is_sinhala_text(sample)
    assert not is_sinhala_text("असमिया कोंकणी English text 123")
    print("[SUCCESS] All self-tests passed.")
