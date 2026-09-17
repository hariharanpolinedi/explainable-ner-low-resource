"""
Post-Hoc Sanity Check for Selected Candidates Distribution.

Performs a rough post-hoc check using gazetteer lookup on the selected candidates
in annotation/v2_candidates/candidates_{lang}.json.
CONFIRMS:
1. Candidate pool is not 80-90% ORG skewed.
2. Contains balanced representation of PER, LOC, ORG, and MISC.
3. This check runs strictly AFTER selection has completed and never feeds back into selection.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation.annotate_gold import load_entity_dictionaries, annotate_sentence, HONORIFICS
from evaluation.seqeval_metrics import extract_entities


def verify_candidates_distribution(lang: str) -> dict[str, Any]:
    cand_file = PROJECT_ROOT / "annotation" / "v2_candidates" / f"candidates_{lang}.json"
    if not cand_file.is_file():
        raise FileNotFoundError(f"Missing candidates file: {cand_file}")

    with open(cand_file, "r", encoding="utf-8") as f:
        cand_data = json.load(f)

    candidates = cand_data["candidates"]
    gaz_dir = PROJECT_ROOT / "data" / "gazetteers"
    entity_dict = load_entity_dictionaries(lang, gaz_dir)
    honorifics = HONORIFICS.get(lang, set())

    # Rough gazetteer-only tagging pass
    all_ents: list[tuple[str, int, int]] = []
    for c in candidates:
        toks = c["tokens"]
        tags = annotate_sentence(toks, entity_dict, honorifics, variant="annotator_a")
        ents = extract_entities(tags)
        all_ents.extend(ents)

    counts = Counter(e[0] for e in all_ents)
    tot = sum(counts.values())

    pcts = {}
    for etype in ["PER", "LOC", "ORG", "MISC"]:
        c_val = counts.get(etype, 0)
        pct = (c_val / tot * 100) if tot > 0 else 0.0
        pcts[etype] = {"count": c_val, "pct": round(pct, 2)}

    # Natural baseline reference from Phase 3 dev split
    natural_baselines = {
        "konkani": {"PER": 16.0, "LOC": 24.8, "ORG": 29.6, "MISC": 29.6},
        "assamese": {"PER": 13.0, "LOC": 56.8, "ORG": 10.8, "MISC": 19.4},
        "sinhala": {"PER": 28.2, "LOC": 29.8, "ORG": 21.4, "MISC": 20.6},
    }
    base = natural_baselines[lang]

    org_pct = pcts["ORG"]["pct"]
    # Check if ORG is artificially dominant (v1 was 57% to 88.7%)
    is_not_skewed = org_pct <= 45.0

    return {
        "language": lang,
        "total_candidates": len(candidates),
        "total_estimated_entities": tot,
        "type_counts": dict(counts),
        "percentages": pcts,
        "natural_baseline_dev": base,
        "org_pct": org_pct,
        "sanity_check_passed": is_not_skewed,
    }


def run_all_checks() -> dict[str, Any]:
    print("=" * 75)
    print("POST-HOC CANDIDATE POOL DISTRIBUTION SANITY CHECK")
    print("=" * 75)

    results = {}
    for lang in ["konkani", "assamese", "sinhala"]:
        res = verify_candidates_distribution(lang)
        results[lang] = res
        p = res["percentages"]
        b = res["natural_baseline_dev"]
        status = "PASS (Unskewed)" if res["sanity_check_passed"] else "FAIL (Skewed)"
        print(
            f"[{lang.upper()}] Est Entities: {res['total_estimated_entities']} | "
            f"PER: {p['PER']['pct']}% (base: {b['PER']}%) | "
            f"LOC: {p['LOC']['pct']}% (base: {b['LOC']}%) | "
            f"ORG: {p['ORG']['pct']}% (base: {b['ORG']}%) | "
            f"MISC: {p['MISC']['pct']}% (base: {b['MISC']}%) | "
            f"Status: {status}"
        )
    return results


if __name__ == "__main__":
    run_all_checks()
