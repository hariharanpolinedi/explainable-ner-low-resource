"""
Assamese Inflectional Suffix Rule Table.

Defines explicit, hand-written rule tables for Assamese noun morphology (case suffixes)
focusing on place names and organizations:
1. Locative (-ত / -অত / -তে)
2. Genitive (-ৰ / -অৰ)
3. Allative (-লৈ / -অলৈ)
4. Ablative (-ৰপৰা)
5. Dative/Accusative (-ক / -অক)

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


def inflect_assamese_noun(lemma: str) -> list[dict[str, Any]]:
    """
    Generate valid inflected forms for an Assamese noun lemma with case metadata.
    """
    results = []

    # Check whether lemma ends with vowel diacritic
    # Assamese vowel diacritics: া (09be), ি (09bf), ী (09c0), ু (09c1), ূ (09c2), ৃ (09c3), ে (09c7), ৈ (09c8), ো (09cb), ৌ (09cc)
    last_char = lemma[-1]
    has_vowel_sign = "\u09be" <= last_char <= "\u09cc"

    # 1. Locative: -ত (e.g. গুৱাহাটীত, অসমত, যোৰহাটত)
    loc_form = lemma + "ত"
    results.append({
        "surface": loc_form,
        "case": "locative",
        "suffix": "-ত",
        "lemma": lemma,
    })

    # 2. Genitive: -ৰ (e.g. অসমৰ, গুৱাহাটীৰ, ভাৰতৰ)
    gen_form = lemma + "ৰ"
    results.append({
        "surface": gen_form,
        "case": "genitive",
        "suffix": "-ৰ",
        "lemma": lemma,
    })

    # 3. Allative (towards): -লৈ (e.g. অসমলৈ, গুৱাহাটীলৈ)
    all_form = lemma + "লৈ"
    results.append({
        "surface": all_form,
        "case": "allative",
        "suffix": "-লৈ",
        "lemma": lemma,
    })

    # 4. Ablative (from): -ৰপৰা (e.g. গুৱাহাটীৰপৰা, অসমৰপৰা)
    abl_form = gen_form + "পৰা"
    results.append({
        "surface": abl_form,
        "case": "ablative",
        "suffix": "-ৰপৰা",
        "lemma": lemma,
    })

    # 5. Dative: -ক (e.g. অসমক, গুৱাহাটীক)
    dat_form = lemma + "ক"
    results.append({
        "surface": dat_form,
        "case": "dative",
        "suffix": "-ক",
        "lemma": lemma,
    })

    return results


if __name__ == "__main__":
    for place in ["গুৱাহাটী", "অসম", "যোৰহাট", "ভাৰত", "মাজুলী"]:
        inflections = inflect_assamese_noun(place)
        print(f"\n{place}:")
        for inf in inflections:
            print(f"  {inf['case']:<15} ({inf['suffix']:<8}) -> {inf['surface']}")
