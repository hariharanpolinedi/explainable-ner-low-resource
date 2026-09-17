"""
Compute PER/LOC/ORG/MISC Counts and Percentages Side-by-Side:
(a) long_multientity_eval.conll
(b) gold-dev
(c) gold-test
(d) gold-train
Per language.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.seqeval_metrics import extract_entities


def parse_conll_entities(file_path: Path) -> dict[str, Any]:
    """Extract entities and sentence/token counts from a CoNLL file."""
    if not file_path.is_file():
        return {"error": f"File not found: {file_path}"}

    sentences = 0
    tokens_count = 0
    current_tags: list[str] = []
    all_entities: list[tuple[str, int, int]] = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tags:
                    sentences += 1
                    ents = extract_entities(current_tags)
                    all_entities.extend(ents)
                    current_tags = []
            else:
                tokens_count += 1
                parts = line.split("\t")
                tag = parts[1] if len(parts) > 1 else parts[0]
                current_tags.append(tag)

    if current_tags:
        sentences += 1
        ents = extract_entities(current_tags)
        all_entities.extend(ents)

    counts = Counter(e[0] for e in all_entities)
    total_ents = sum(counts.values())

    percs = {}
    for cat in ["PER", "LOC", "ORG", "MISC"]:
        cnt = counts.get(cat, 0)
        pct = (cnt / total_ents * 100.0) if total_ents > 0 else 0.0
        percs[cat] = {"count": cnt, "pct": round(pct, 2)}

    return {
        "file": str(file_path.relative_to(PROJECT_ROOT)),
        "sentences": sentences,
        "tokens": tokens_count,
        "total_entities": total_ents,
        "entity_counts": dict(counts),
        "percentages": percs,
    }


def compare_all_distributions() -> dict[str, Any]:
    languages = ["konkani", "assamese", "sinhala"]
    splits = [
        ("long_multientity_eval", "long_multientity_eval.conll"),
        ("dev", "dev.conll"),
        ("test", "test.conll"),
        ("train", "train.conll"),
    ]

    all_results: dict[str, dict[str, Any]] = {}

    for lang in languages:
        all_results[lang] = {}
        for split_name, fname in splits:
            fpath = PROJECT_ROOT / "data" / "gold" / lang / fname
            all_results[lang][split_name] = parse_conll_entities(fpath)

    return all_results


def print_comparison_tables(results: dict[str, Any]) -> None:
    for lang, splits in results.items():
        print(f"\n==================== {lang.upper()} ====================")
        print(f"{'Split':<24} | {'Sentences':<9} | {'Total Ents':<10} | {'PER (%)':<16} | {'LOC (%)':<16} | {'ORG (%)':<16} | {'MISC (%)':<16}")
        print("-" * 115)
        for sname, data in splits.items():
            if "error" in data:
                print(f"{sname:<24} | ERROR: {data['error']}")
                continue
            sents = data["sentences"]
            tents = data["total_entities"]
            p = data["percentages"]
            per_str = f"{p['PER']['count']} ({p['PER']['pct']}%)"
            loc_str = f"{p['LOC']['count']} ({p['LOC']['pct']}%)"
            org_str = f"{p['ORG']['count']} ({p['ORG']['pct']}%)"
            misc_str = f"{p['MISC']['count']} ({p['MISC']['pct']}%)"
            print(f"{sname:<24} | {sents:<9} | {tents:<10} | {per_str:<16} | {loc_str:<16} | {org_str:<16} | {misc_str:<16}")


if __name__ == "__main__":
    res = compare_all_distributions()
    print_comparison_tables(res)
