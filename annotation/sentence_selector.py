"""
Entity-dense sentence selection pipeline for gold NER annotation.

Prioritizes sentences containing gazetteer matches (PER, LOC, ORG)
to prevent class starvation and ensure rich representation of rare entity types.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import BaseConfig, load_config, parse_cli_args
from utils.logger import ExperimentLogger
from utils.seed import set_global_seed


def load_gazetteers(lang: str, gazetteer_dir: Path) -> dict[str, set[str]]:
    """Load person, place, org, and misc gazetteer sets for a language."""
    gaz_data: dict[str, set[str]] = {"persons": set(), "places": set(), "orgs": set(), "misc": set()}
    lang_dir = gazetteer_dir / lang

    for cat in ["persons", "places", "orgs"]:
        fpath = lang_dir / f"{cat}.txt"
        if fpath.is_file():
            with open(fpath, "r", encoding="utf-8") as fp:
                for line in fp:
                    item = line.strip()
                    if item:
                        gaz_data[cat].add(item)

    # Add cultural/statutory MISC gazetteer terms per language
    if lang == "konkani":
        gaz_data["misc"] = {
            "शिगमो", "चवथ", "दिवाळी", "नाताल", "साहित्य अकादेमी पुरस्कार", "अकादेमी पुरस्कार",
            "गोंय मुक्ती दीस", "ज्ञानपीठ पुरस्कार", "पद्मश्री", "पद्मभूषण", "संविधान", "रामायण", "महाभारत"
        }
    elif lang == "assamese":
        gaz_data["misc"] = {
            "ৰঙালী বিহু", "ভোগালী বিহু", "কঙালী বিহু", "বিহু", "পদ্মশ্ৰী", "পদ্মভূষণ",
            "জ্ঞানপীঠ বঁটা", "অসম চুক্তি", "বুঢ়ী আইৰ সাধু", "সংবিধান", "ৰামায়ণ", "মহাভাৰত"
        }
    else:
        gaz_data["misc"] = {
            "මහාවංශය", "සිංහල අලුත් අවුරුද්ද", "අවුරුද්ද", "දළදා පෙරහැර", "පෙරහැර", "වෙසක්",
            "පොසොන්", "රණවිරු සම්මානය", "ආණ්ඩුක්‍රම ව්‍යවස්ථාව", "රාමායණය", "මහාභාරතය", "ත්‍රිපිටකය"
        }
    return gaz_data


def score_sentence(tokens: list[str], gazetteers: dict[str, set[str]]) -> tuple[int, set[str]]:
    """
    Score a sentence by counting entity category matches.
    Returns (score, matched_categories).
    """
    sentence_text = " ".join(tokens)
    matched_cats: set[str] = set()
    total_matches = 0

    for cat, items in gazetteers.items():
        for item in items:
            if item in sentence_text:
                matched_cats.add(cat)
                total_matches += 1

    # Bonus score for multi-category diversity (e.g., ORG + LOC or PER + ORG)
    score = total_matches + (len(matched_cats) * 3)
    return score, matched_cats


def select_entity_dense_sentences(
    lang: str,
    processed_dir: Path,
    gazetteer_dir: Path,
    target_count: int = 1000,
    min_tokens: int = 5,
    max_tokens: int = 45,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Select target_count sentences biased toward entity density and category balance.
    Ensures PER, ORG, and MISC are well-represented alongside LOC.
    """
    sentences_file = processed_dir / lang / "sentences.txt"
    if not sentences_file.is_file():
        raise FileNotFoundError(f"Missing sentences file: {sentences_file}")

    gazetteers = load_gazetteers(lang, gazetteer_dir)
    print(
        f"[{lang.upper()}] Loaded gazetteers: {len(gazetteers['persons'])} PER, "
        f"{len(gazetteers['places'])} LOC, {len(gazetteers['orgs'])} ORG, {len(gazetteers['misc'])} MISC"
    )

    candidates: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    with open(sentences_file, "r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            line_str = line.strip()
            if not line_str:
                continue

            tokens = line_str.split()
            if not (min_tokens <= len(tokens) <= max_tokens):
                continue

            if line_str in seen_hashes:
                continue
            seen_hashes.add(line_str)

            score, cats = score_sentence(tokens, gazetteers)
            candidates.append({
                "sentence": line_str,
                "tokens": tokens,
                "token_count": len(tokens),
                "score": score,
                "matched_categories": list(cats),
            })

    # Partition candidates by entity category to ensure balanced representation
    rng = random.Random(seed)
    
    per_cands = [c for c in candidates if "persons" in c["matched_categories"]]
    org_cands = [c for c in candidates if "orgs" in c["matched_categories"]]
    misc_cands = [c for c in candidates if "misc" in c["matched_categories"]]
    multi_cands = [c for c in candidates if len(c["matched_categories"]) >= 2]
    loc_only_cands = [c for c in candidates if c["matched_categories"] == ["places"]]
    neutral_cands = [c for c in candidates if c["score"] == 0]

    rng.shuffle(per_cands)
    rng.shuffle(org_cands)
    rng.shuffle(misc_cands)
    rng.shuffle(multi_cands)
    rng.shuffle(loc_only_cands)
    rng.shuffle(neutral_cands)

    selected: list[dict[str, Any]] = []
    selected_set: set[str] = set()

    def add_from(cands: list[dict[str, Any]], limit: int) -> None:
        count = 0
        for c in cands:
            if c["sentence"] not in selected_set:
                selected.append(c)
                selected_set.add(c["sentence"])
                count += 1
                if count >= limit:
                    break

    # 1. Multi-entity sentences first (high value, diverse)
    add_from(multi_cands, 250)
    # 2. Rare categories: PER, ORG, MISC
    add_from(per_cands, 150)
    add_from(org_cands, 200)
    add_from(misc_cands, 150)
    # 3. LOC sentences
    add_from(loc_only_cands, 150)
    # 4. Neutral natural sentences (10% natural background)
    add_from(neutral_cands, 100)

    # If shortfall remains, fill with remaining candidates sorted by score
    if len(selected) < target_count:
        remaining = [c for c in candidates if c["sentence"] not in selected_set]
        remaining.sort(key=lambda x: x["score"], reverse=True)
        add_from(remaining, target_count - len(selected))

    # Ensure the first 100 sentences (the overlap sample) contain a rich mix of all 4 types
    overlap_sample: list[dict[str, Any]] = []
    overlap_set: set[str] = set()
    for cat in ["persons", "orgs", "misc", "places"]:
        cat_items = [c for c in selected if cat in c["matched_categories"] and c["sentence"] not in overlap_set]
        for c in cat_items[:22]:
            overlap_sample.append(c)
            overlap_set.add(c["sentence"])

    # Fill remainder of overlap sample up to 100 with diverse sentences
    for c in selected:
        if len(overlap_sample) >= 100:
            break
        if c["sentence"] not in overlap_set:
            overlap_sample.append(c)
            overlap_set.add(c["sentence"])

    # Reconstruct selected so overlap_sample is the first 100 sentences
    non_overlap = [c for c in selected if c["sentence"] not in overlap_set]
    final_selected = overlap_sample + non_overlap

    print(f"[{lang.upper()}] Final selected: {len(final_selected)} sentences. Overlap mix: {len(overlap_sample)} sentences.")
    return final_selected[:target_count]


def run_sentence_selection(config_path: str = "configs/base_config.yaml") -> dict[str, Any]:
    cfg: BaseConfig = load_config(config_path)
    set_global_seed(cfg.seed)

    logger = ExperimentLogger(log_dir=cfg.logging.log_dir, backend=cfg.logging.backend)
    processed_dir = PROJECT_ROOT / "data" / "processed"
    gazetteer_dir = PROJECT_ROOT / "data" / "gazetteers"
    annotation_dir = PROJECT_ROOT / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    target_count = cfg.annotation.target_sentence_count if cfg.annotation else 1000
    results: dict[str, Any] = {}

    for lang in cfg.project.languages:
        selected = select_entity_dense_sentences(
            lang=lang,
            processed_dir=processed_dir,
            gazetteer_dir=gazetteer_dir,
            target_count=target_count,
            seed=cfg.seed,
        )
        out_file = annotation_dir / f"selected_{lang}.json"
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump(selected, fp, ensure_ascii=False, indent=2)

        results[f"{lang}_selected_count"] = len(selected)
        avg_len = sum(s["token_count"] for s in selected) / len(selected) if selected else 0
        results[f"{lang}_avg_token_length"] = round(avg_len, 2)

    logger.log_run(
        config_path=config_path,
        metrics=results,
        dataset_version="v0.3-selected-annotation-pool",
    )
    return results


if __name__ == "__main__":
    cfg_file, _ = parse_cli_args(sys.argv[1:])
    run_sentence_selection(cfg_file)
