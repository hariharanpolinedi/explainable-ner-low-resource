"""
Assamese Inflectional OOV Gap Verification Script.

Scans Assamese gold data (data/gold/assamese/train.conll and dev.conll) and tests
the deployed Assamese NER predictor on inflected place names vs lemma forms.

Identifies:
1. Frequency of gazetteer place stems occurring with case inflection suffixes (-ত, -ৰ, -লৈ, -ৰপৰা, -ক).
2. Gold tag distribution of these inflected forms (documenting whether they were omitted as 'O').
3. Empirical behavior of checkpoints/augmented_assamese.pt on inflected vs uninflected sentences.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.inference import get_predictor


def load_gazetteer(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def scan_conll_for_inflected_stems(
    conll_path: Path,
    stems: list[str],
) -> list[dict[str, Any]]:
    results = []
    lines = conll_path.read_text(encoding="utf-8").splitlines()
    for line_idx, line in enumerate(lines, 1):
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            tok, tag = parts[0], parts[1]
            for stem in stems:
                if len(stem) >= 3 and tok.startswith(stem) and len(tok) > len(stem):
                    suffix = tok[len(stem):]
                    # Filter for known case/morpheme suffixes in Assamese
                    if any(suffix.startswith(s) for s in ["ত", "ৰ", "লৈ", "ক", "ে", "ৰপৰা", "ীয়"]):
                        results.append({
                            "token": tok,
                            "stem": stem,
                            "suffix": suffix,
                            "tag": tag,
                            "line": line_idx,
                        })
    return results


def run_assamese_inflection_gap_audit() -> dict[str, Any]:
    print("=" * 80)
    print("ASSAMESE INFLECTIONAL GAP VERIFICATION AUDIT")
    print("=" * 80)

    gold_dir = PROJECT_ROOT / "data" / "gold" / "assamese"
    gaz_places = load_gazetteer(PROJECT_ROOT / "data" / "gazetteers" / "assamese" / "places.txt")
    print(f"Loaded {len(gaz_places)} Assamese place names from gazetteer.")

    train_matches = scan_conll_for_inflected_stems(gold_dir / "train.conll", gaz_places)
    dev_matches = scan_conll_for_inflected_stems(gold_dir / "dev.conll", gaz_places)

    print(f"\n1. In Gold Train ({gold_dir / 'train.conll'}):")
    print(f"   Total inflected gazetteer occurrences: {len(train_matches)}")
    train_suffixes = Counter(m["suffix"] for m in train_matches)
    train_tags = Counter(m["tag"] for m in train_matches)
    print(f"   Top suffixes: {train_suffixes.most_common(6)}")
    print(f"   Tag distribution: {dict(train_tags)}")

    print(f"\n2. In Gold Dev ({gold_dir / 'dev.conll'}):")
    print(f"   Total inflected gazetteer occurrences: {len(dev_matches)}")
    dev_suffixes = Counter(m["suffix"] for m in dev_matches)
    dev_tags = Counter(m["tag"] for m in dev_matches)
    print(f"   Top suffixes: {dev_suffixes.most_common(6)}")
    print(f"   Tag distribution: {dict(dev_tags)}")

    # Check how many inflected occurrences are tagged as LOC
    train_loc_inflected = [m for m in train_matches if m["tag"] in ("B-LOC", "I-LOC")]
    dev_loc_inflected = [m for m in dev_matches if m["tag"] in ("B-LOC", "I-LOC")]
    print(f"\n   Inflected forms tagged as B-LOC / I-LOC in Train: {len(train_loc_inflected)} / {len(train_matches)} ({len(train_loc_inflected)/max(1, len(train_matches))*100:.1f}%)")
    print(f"   Inflected forms tagged as B-LOC / I-LOC in Dev:   {len(dev_loc_inflected)} / {len(dev_matches)} ({len(dev_loc_inflected)/max(1, len(dev_matches))*100:.1f}%)")

    # Predictor empirical test
    print("\n3. Live Inference Test on Assamese Lemma vs Inflected Forms:")
    predictor = get_predictor("assamese")
    test_cases = [
        ("Lemma (Guwahati)", "মই গুৱাহাটী বাস কৰো।", "গুৱাহাটী"),
        ("Locative -ত (Guwahati-t)", "মই গুৱাহাটীত বাস কৰো।", "গুৱাহাটীত"),
        ("Lemma (Assam)", "অসম এখন সুন্দৰ ৰাজ্য।", "অসম"),
        ("Genitive -ৰ (Assam-or)", "অসমৰ ৰাজধানী দিছপুৰ।", "অসমৰ"),
        ("Lemma (Jorhat)", "তেওঁ যোৰহাট থাকে।", "যোৰহাট"),
        ("Locative -ত (Jorhat-ot)", "তেওঁ যোৰহাটত থাকে।", "যোৰহাটত"),
        ("Allative -লৈ (Assam-loi)", "তেওঁ অসমলৈ উভতি আহিল।", "অসমলৈ"),
    ]

    inference_results = []
    for label, text, target in test_cases:
        pred = predictor.predict(text)
        ents = [(e["text"], e["type"]) for e in pred.get("entities", [])]
        found = any(target in e[0] for e in ents)
        status = "RECOVERED" if found else "MISSED (OOV/Collapse)"
        print(f"   [{label}] '{text}' -> Entities: {ents} | Target '{target}': {status}")
        inference_results.append({
            "label": label,
            "text": text,
            "target": target,
            "entities": ents,
            "found": found,
        })

    is_gap_present = (
        len(train_loc_inflected) == 0
        and not any(r["found"] for r in inference_results if "Locative" in r["label"] or "Genitive" in r["label"])
    )

    print("\n" + "=" * 80)
    print(f"ASSAMESE INFLECTION GAP STATUS: {'CONFIRMED PRESENT' if is_gap_present else 'ABSENT'}")
    print("=" * 80)

    return {
        "is_gap_present": is_gap_present,
        "train_matches_count": len(train_matches),
        "dev_matches_count": len(dev_matches),
        "train_loc_inflected_count": len(train_loc_inflected),
        "inference_results": inference_results,
    }


if __name__ == "__main__":
    run_assamese_inflection_gap_audit()
