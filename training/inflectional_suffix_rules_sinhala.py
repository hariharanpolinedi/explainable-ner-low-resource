"""
Sinhala Inflectional Suffix Rule Table.

Defines explicit, hand-written rule tables for Sinhala noun morphology (case suffixes)
focusing on place names, organizations, and persons:
1. Locative (-යේදී / -දී / -හි / -තුළ)
2. Genitive (-වේ / -යේ / -ගේ / -හි)
3. Dative (-ට / -යට)
4. Instrumental/Ablative (-න් / -ෙන් / -ගෙන් / -මගින්)

Zero pretrained morphological analyzers or external packages.
"""

from __future__ import annotations

import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def inflect_sinhala_noun(lemma: str, entity_type: str = "LOC") -> list[dict[str, Any]]:
    """
    Generate valid inflected forms for a Sinhala noun lemma with case metadata.
    Handles vowel-ending stems:
    - Words ending in -ය (e.g. විශ්වවිද්‍යාලය, පේරාදෙණිය):
      stem drops ය or adds vowel diacritic e/e: -> -යේදී, -යේ, -යට, -යෙන්
    - Words ending in -ව (e.g. ලංකාව):
      -> -වේ, -වේදී, -වට, -වෙන්
    - Words ending in inherent/other consonants/vowels (e.g. කොළඹ):
      -> -දී, -ට, -ින්, -හි
    """
    results = []

    if lemma.endswith("ය"):
        # e.g., විශ්වවිද්‍යාලය -> stem = විශ්වවිද්‍යාල
        base = lemma[:-1]
        loc_suf = "යේදී"
        gen_suf = "යේ"
        dat_suf = "යට"
        abl_suf = "යෙන්"

        loc_form = base + loc_suf
        gen_form = base + gen_suf
        dat_form = base + dat_suf
        abl_form = base + abl_suf

    elif lemma.endswith("ව"):
        # e.g., ලංකාව -> base = ලංකා
        base = lemma[:-1]
        loc_suf = "වේදී"
        gen_suf = "වේ"
        dat_suf = "වට"
        abl_suf = "වෙන්"

        loc_form = base + loc_suf
        gen_form = base + gen_suf
        dat_form = base + dat_suf
        abl_form = base + abl_suf

    else:
        # e.g., කොළඹ, මහනුවර, යාපනය
        loc_form = lemma + "දී"
        gen_form = lemma + "හි"
        dat_form = lemma + "ට"
        abl_form = lemma + "ින්"

    # 1. Locative
    results.append({
        "surface": loc_form,
        "case": "locative",
        "suffix": "-යේදී/-දී",
        "lemma": lemma,
    })

    # 2. Genitive
    results.append({
        "surface": gen_form,
        "case": "genitive",
        "suffix": "-වේ/-යේ",
        "lemma": lemma,
    })

    # 3. Dative
    results.append({
        "surface": dat_form,
        "case": "dative",
        "suffix": "-ට",
        "lemma": lemma,
    })

    # 4. Ablative
    results.append({
        "surface": abl_form,
        "case": "ablative",
        "suffix": "-න්/-ෙන්",
        "lemma": lemma,
    })

    # If entity is Person, also add -ගේ (animate genitive)
    if entity_type == "PER" or lemma in ["අනුර", "සිරිමාවෝ", "මහින්ද", "රනිල්"]:
        results.append({
            "surface": lemma + "ගේ",
            "case": "genitive_animate",
            "suffix": "-ගේ",
            "lemma": lemma,
        })

    return results


if __name__ == "__main__":
    for place in ["විශ්වවිද්‍යාලය", "පේරාදෙණිය", "ලංකාව", "කොළඹ", "ශ්‍රී ලංකාව"]:
        inflections = inflect_sinhala_noun(place)
        print(f"\n{place}:")
        for inf in inflections:
            print(f"  {inf['case']:<18} ({inf['suffix']:<10}) -> {inf['surface']}")
