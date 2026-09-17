"""
Konkani Inflectional Suffix Rule Table.

Defines explicit, hand-written rule tables for Konkani noun morphology (case suffixes)
focusing on place names and organizations:
1. Locative (-ंत / -ांत)
2. Genitive (-चो / -ची / -चे / -च्या / -च्यान)
3. Dative (-क / -ाक)
4. Instrumental/Ablative (-न / -ान / -सून / -थावून / -ंतल्यान)

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

# Specific irregular or standard oblique stems for common Konkani entities
KONKANI_OBLIQUE_STEMS: dict[str, str] = {
    "गोवा": "गोया",
    "मडगांव": "मडगांवा",
    "कुडचडें": "कुडचड्या",
    "वास्को": "वास्को",
    "मुरगांव": "मुरगांवा",
    "भारत": "भारता",
    "महाराष्ट्र": "महाराष्ट्रा",
    "कर्नाटक": "कर्नाटका",
    "केरळ": "केरळा",
}


def get_konkani_oblique_stem(lemma: str) -> str:
    """Return the oblique base for suffix attachment."""
    if lemma in KONKANI_OBLIQUE_STEMS:
        return KONKANI_OBLIQUE_STEMS[lemma]
    # General rule: if ending in consonant (no virama, no vowel diacritic), add matra -ा
    last_char = lemma[-1]
    # Indic vowel diacritics in Devanagari: \u093e to \u094c, \u094d
    if "\u093e" <= last_char <= "\u094c" or last_char == "\u094d":
        return lemma
    # Otherwise ends in inherent 'a', oblique adds 'aa' matra: ा
    return lemma + "ा"


def inflect_konkani_noun(lemma: str) -> list[dict[str, Any]]:
    """
    Generate valid inflected forms for a Konkani noun lemma with case metadata.
    """
    oblique = get_konkani_oblique_stem(lemma)
    results = []

    # 1. Locative: -ंत (in, at)
    # If lemma ends in -ी or -ू or -ो, suffix -ंत attaches directly
    if lemma.endswith("ी") or lemma.endswith("ू") or lemma.endswith("ो"):
        loc_form = lemma + "ंत"
    else:
        loc_form = oblique + "ंत"
    results.append({
        "surface": loc_form,
        "case": "locative",
        "suffix": "-ंत",
        "lemma": lemma,
    })

    # 2. Genitive: -च्या (oblique genitive: "of X", most frequent in text), -चो, -ची, -चे
    # Attaches to oblique stem
    gen_forms = [
        (oblique + "च्या", "-च्या", "genitive_oblique"),
        (oblique + "चो", "-चो", "genitive_masc"),
        (oblique + "ची", "-ची", "genitive_fem"),
        (oblique + "चे", "-चे", "genitive_neut_pl"),
    ]
    for gf, suf, ctype in gen_forms:
        results.append({
            "surface": gf,
            "case": ctype,
            "suffix": suf,
            "lemma": lemma,
        })

    # 3. Dative: -क (to X, for X)
    dat_form = (lemma + "क") if (lemma.endswith("ी") or lemma.endswith("ू")) else (oblique + "क")
    results.append({
        "surface": dat_form,
        "case": "dative",
        "suffix": "-क",
        "lemma": lemma,
    })

    # 4. Ablative/Origin: -ंतल्यान (from inside X), -सून (from X)
    results.append({
        "surface": loc_form + "ल्यान",
        "case": "locative_ablative",
        "suffix": "-ंतल्यान",
        "lemma": lemma,
    })
    results.append({
        "surface": lemma + "सून",
        "case": "ablative",
        "suffix": "-सून",
        "lemma": lemma,
    })

    return results


if __name__ == "__main__":
    for place in ["पणजी", "गोवा", "मडगांव", "भारत", "दिल्ली"]:
        inflections = inflect_konkani_noun(place)
        print(f"\n{place}:")
        for inf in inflections:
            print(f"  {inf['case']:<18} ({inf['suffix']:<8}) -> {inf['surface']}")
