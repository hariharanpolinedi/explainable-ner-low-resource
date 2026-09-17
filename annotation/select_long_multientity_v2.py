"""
Length and Structure-Only Candidate Sentence Selection (v2).

Selects candidates from data/processed/{lang}/sentences.txt:
- Length: configurable min_tokens to max_tokens (default: 30 to 80)
- Syntactic Delimiter: configurable min_delimiters (default: >= 2 commas or conjunctions)
- Strictly non-overlapping with existing Gold (train/dev/test), Silver, Pseudo, Augmented splits.
- Zero gazetteer matching, zero rule taggers, zero entity-aware filters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONJUNCTIONS: dict[str, set[str]] = {
    "konkani": {",", "आनी", "आणि", "वा", "तशेंच", ";"},
    "assamese": {",", "আৰু", "বা", "তেনেদৰে", "লগতে", ";"},
    "sinhala": {",", "සහ", "හා", "හෝ", "මෙන්ම", ";"},
}


def load_used_sentences(lang: str) -> set[str]:
    """Load normalized text of all existing sentences across all project splits to prevent data leakage."""
    used: set[str] = set()
    base_dirs = [
        PROJECT_ROOT / "data" / "gold" / lang,
        PROJECT_ROOT / "data" / "silver" / lang,
        PROJECT_ROOT / "data" / "pseudo" / lang,
        PROJECT_ROOT / "data" / "augmented" / lang,
    ]

    for bdir in base_dirs:
        if not bdir.is_dir():
            continue
        for p in bdir.glob("*.conll"):
            if "long_multientity" in p.name:
                continue
            curr: list[str] = []
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        if curr:
                            used.add(" ".join(curr))
                            curr = []
                    else:
                        parts = line.split("\t")
                        curr.append(parts[0])
            if curr:
                used.add(" ".join(curr))
    return used


def count_list_delimiters(tokens: list[str], lang: str) -> int:
    """Count occurrences of comma and native conjunction tokens in sequence."""
    delims = CONJUNCTIONS.get(lang, {","})
    return sum(1 for t in tokens if t in delims or "," in t)


def select_candidates_for_language(
    lang: str,
    target_count: int = 40,
    min_tokens: int = 30,
    max_tokens: int = 80,
    min_delimiters: int = 2,
) -> dict[str, Any]:
    raw_file = PROJECT_ROOT / "data" / "processed" / lang / "sentences.txt"
    if not raw_file.is_file():
        raise FileNotFoundError(f"Raw sentences file not found: {raw_file}")

    used_sentences = load_used_sentences(lang)
    selected: list[dict[str, Any]] = []
    seen_in_run: set[str] = set()

    total_long_sentences = 0
    total_matching_structure = 0

    with open(raw_file, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, start=1):
            s_text = line.strip()
            if not s_text or s_text in used_sentences or s_text in seen_in_run:
                continue

            toks = s_text.split()
            token_count = len(toks)

            if token_count < min_tokens or token_count > max_tokens:
                continue
            total_long_sentences += 1

            delim_count = count_list_delimiters(toks, lang)
            if delim_count < min_delimiters:
                continue
            total_matching_structure += 1

            seen_in_run.add(s_text)
            selected.append({
                "id": f"{lang}_cand_{len(selected) + 1:03d}",
                "line_num": line_num,
                "token_count": token_count,
                "delimiter_count": delim_count,
                "tokens": toks,
                "text": s_text,
            })

            if len(selected) >= target_count:
                break

    return {
        "language": lang,
        "target_count": target_count,
        "selected_count": len(selected),
        "total_long_sentences_scanned": total_long_sentences,
        "total_matching_structure": total_matching_structure,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "min_delimiters": min_delimiters,
        "candidates": selected,
    }


def select_all_candidates(
    target_count: int = 40,
    min_tokens: int = 30,
    max_tokens: int = 80,
    min_delimiters: int = 2,
) -> dict[str, Any]:
    results = {}
    out_dir = PROJECT_ROOT / "annotation" / "v2_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("LENGTH & STRUCTURE-ONLY CANDIDATE SELECTION (v2)")
    print(f"Token range: [{min_tokens}, {max_tokens}], Min list delimiters: {min_delimiters}")
    print("=" * 75)

    for lang in ["konkani", "assamese", "sinhala"]:
        res = select_candidates_for_language(
            lang=lang,
            target_count=target_count,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            min_delimiters=min_delimiters,
        )
        results[lang] = res

        out_path = out_dir / f"candidates_{lang}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        print(
            f"[{lang.upper()}] Selected {res['selected_count']} / {target_count} candidates "
            f"(from {res['total_matching_structure']} qualifying sentences, "
            f"{res['total_long_sentences_scanned']} long sentences scanned) -> {out_path.name}"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Length and structure-only candidate selector for v2 evaluation set")
    parser.add_argument("--target-count", type=int, default=40, help="Number of sentences to select per language (default: 40)")
    parser.add_argument("--min-tokens", type=int, default=30, help="Minimum tokens per sentence (default: 30)")
    parser.add_argument("--max-tokens", type=int, default=80, help="Maximum tokens per sentence (default: 80)")
    parser.add_argument("--min-delimiters", type=int, default=2, help="Minimum comma/conjunction delimiters (default: 2)")
    args = parser.parse_args()

    select_all_candidates(
        target_count=args.target_count,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        min_delimiters=args.min_delimiters,
    )


if __name__ == "__main__":
    main()
