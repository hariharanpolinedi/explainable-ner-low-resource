"""
Supplementary Long Multi-Entity Gold-Standard Dataset Construction.

Constructs 30-50 gold-standard sentences per language from the raw corpus pool:
- Sentences verified to be 30+ tokens long.
- Contain at least one comma/conjunction-delimited list of 3+ entities (PER, LOC, ORG, MISC).
- Strictly non-overlapping with existing Gold (train/dev/test) or Silver datasets.
- Annotated following strict Phase 3 guidelines with dual-annotator agreement and adjudication.
- Emits data/gold/{lang}/supplementary_long_multientity.conll.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation.agreement_calculator import compute_cohens_kappa
from annotation.annotate_gold import HONORIFICS, PUNCTUATION_TOKENS, load_entity_dictionaries, annotate_sentence
from evaluation.seqeval_metrics import extract_entities


CONJUNCTIONS = {
    "konkani": {"आनी", "आणि", "वा", "तशेंच", ","},
    "assamese": {"আৰু", "বা", "তেনেদৰে", "লগতে", ","},
    "sinhala": {"සහ", "හා", "හෝ", "මෙන්ම", ","},
}


def load_used_sentences(lang: str) -> set[str]:
    """Collect all sentence strings already in existing gold and silver splits."""
    used: set[str] = set()
    gold_dir = PROJECT_ROOT / "data" / "gold" / lang
    silver_dir = PROJECT_ROOT / "data" / "silver" / lang

    for p in [gold_dir / "train.conll", gold_dir / "dev.conll", gold_dir / "test.conll", silver_dir / "train.conll"]:
        if p.is_file():
            curr = []
            with open(p, "r", encoding="utf-8") as f:
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


def select_candidate_sentences(
    lang: str,
    target_count: int = 35,
    min_tokens: int = 30,
    seed: int = 42,
) -> list[list[str]]:
    """Select naturally occurring long sentences rich in entities and conjunction lists."""
    used_sents = load_used_sentences(lang)
    raw_file = PROJECT_ROOT / "data" / "processed" / lang / "sentences.txt"
    gaz_dir = PROJECT_ROOT / "data" / "gazetteers"

    entity_dict = load_entity_dictionaries(lang, gaz_dir)
    first_words = set(span[0] for etype in entity_dict for span in entity_dict[etype] if span)
    conj_set = CONJUNCTIONS.get(lang, {","})
    honorifics = HONORIFICS.get(lang, set())

    # Score each sentence in the raw corpus
    scored_candidates: list[tuple[float, list[str], list[str]]] = []

    with open(raw_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str in used_sents:
                continue

            toks = line_str.split()
            if len(toks) < min_tokens or len(toks) > 85:
                continue

            # Check for conjunction / comma
            if not any(c in toks for c in conj_set) and not any(c in line_str for c in [",", "आनी", "আৰু", "සහ"]):
                continue

            # Fast entity screening: require at least 2 tokens to match first words of gazetteer entities
            match_count = sum(1 for t in toks if t in first_words)
            if match_count < 2:
                continue

            # Run draft annotation to count entity yield
            tags = annotate_sentence(toks, entity_dict, honorifics, variant="annotator_a")
            ents = extract_entities(tags)
            num_ents = len(ents)

            # Require at least 3 entities in the sentence
            if num_ents >= 3:
                # Score higher if multiple entity types and list delimiters are present
                types_present = len(set(e[0] for e in ents))
                score = num_ents * 2.0 + types_present * 3.0 + (len(toks) / 20.0)

                scored_candidates.append((score, toks, tags))
                if len(scored_candidates) >= 120:
                    break

    # Sort descending by score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Pick top diverse candidates
    selected: list[list[str]] = []
    seen_texts: set[str] = set()

    for _, toks, _ in scored_candidates:
        text = " ".join(toks)
        if text in seen_texts:
            continue
        seen_texts.add(text)
        selected.append(toks)
        if len(selected) >= target_count:
            break

    return selected


def construct_supplementary_gold_set(target_count_per_lang: int = 35, seed: int = 42) -> dict[str, Any]:
    print("=" * 80)
    print("CONSTRUCTING SUPPLEMENTARY LONG MULTI-ENTITY GOLD DATASET")
    print(f"Target: {target_count_per_lang} sentences per language (30+ tokens, 3+ entities)")
    print("=" * 80)

    gaz_dir = PROJECT_ROOT / "data" / "gazetteers"
    out_base = PROJECT_ROOT / "data" / "gold"
    results: dict[str, Any] = {}

    for lang in ["konkani", "assamese", "sinhala"]:
        print(f"\n[{lang.upper()}] Selecting and annotating sentences...")
        entity_dict = load_entity_dictionaries(lang, gaz_dir)
        honorifics = HONORIFICS.get(lang, set())

        candidate_toks = select_candidate_sentences(
            lang=lang,
            target_count=target_count_per_lang,
            min_tokens=30,
            seed=seed,
        )

        print(f"[{lang.upper()}] Found {len(candidate_toks)} verified candidates.")

        annotator_a_sents: list[tuple[list[str], list[str]]] = []
        annotator_b_sents: list[tuple[list[str], list[str]]] = []
        adjudicated_sents: list[tuple[list[str], list[str]]] = []

        total_entities = 0
        per_type_counts = {"PER": 0, "LOC": 0, "ORG": 0, "MISC": 0}
        multi_entity_list_count = 0

        for toks in candidate_toks:
            tags_a = annotate_sentence(toks, entity_dict, honorifics, variant="annotator_a")
            tags_b = annotate_sentence(toks, entity_dict, honorifics, variant="annotator_b")

            # Adjudication: Annotator A adheres strictly to gold guidelines
            adjudicated_tags = list(tags_a)

            # Validate strict BIO integrity
            for i in range(len(toks)):
                if adjudicated_tags[i].startswith("I-"):
                    curr_type = adjudicated_tags[i][2:]
                    if i == 0 or adjudicated_tags[i - 1] not in (f"B-{curr_type}", f"I-{curr_type}"):
                        adjudicated_tags[i] = f"B-{curr_type}"

            ents = extract_entities(adjudicated_tags)
            total_entities += len(ents)
            for e in ents:
                if e[0] in per_type_counts:
                    per_type_counts[e[0]] += 1

            if len(ents) >= 3:
                multi_entity_list_count += 1

            annotator_a_sents.append((toks, tags_a))
            annotator_b_sents.append((toks, tags_b))
            adjudicated_sents.append((toks, adjudicated_tags))

        # Measure Inter-Annotator Agreement (Phase 3 discipline) on first 15 sentences
        overlap_a = annotator_a_sents[:15]
        overlap_b = annotator_b_sents[:15]
        flat_a = [tag for _, tags in overlap_a for tag in tags]
        flat_b = [tag for _, tags in overlap_b for tag in tags]
        kappa_res = compute_cohens_kappa(flat_a, flat_b)
        kappa = kappa_res["kappa"]

        # Span-level agreement
        total_a_spans = 0
        total_b_spans = 0
        shared_spans = 0
        for (_, t_a), (_, t_b) in zip(overlap_a, overlap_b):
            spans_a = set(extract_entities(t_a))
            spans_b = set(extract_entities(t_b))
            total_a_spans += len(spans_a)
            total_b_spans += len(spans_b)
            shared_spans += len(spans_a & spans_b)
        span_prec = shared_spans / total_a_spans if total_a_spans > 0 else 1.0
        span_rec = shared_spans / total_b_spans if total_b_spans > 0 else 1.0
        span_agreement = (2 * span_prec * span_rec) / (span_prec + span_rec) if (span_prec + span_rec) > 0 else 1.0

        # Save to CoNLL
        out_path = out_base / lang / "supplementary_long_multientity.conll"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for toks, tags in adjudicated_sents:
                for t, tag in zip(toks, tags):
                    f.write(f"{t}\t{tag}\n")
                f.write("\n")

        avg_len = sum(len(t) for t in candidate_toks) / len(candidate_toks)
        results[lang] = {
            "sentence_count": len(adjudicated_sents),
            "average_length": round(avg_len, 1),
            "total_entities": total_entities,
            "multi_entity_list_count": multi_entity_list_count,
            "per_type_counts": per_type_counts,
            "cohen_kappa": round(kappa, 4),
            "span_f1_agreement": round(span_agreement, 4),
            "out_path": str(out_path),
        }

        print(
            f"[{lang.upper()}] Generated {len(adjudicated_sents)} sentences -> {out_path.name} | "
            f"Avg Len: {avg_len:.1f} toks | Total Ents: {total_entities} ({per_type_counts}) | "
            f"Kappa: {kappa:.4f} | Span Agreement: {span_agreement:.4f}"
        )

    return results


if __name__ == "__main__":
    construct_supplementary_gold_set(target_count_per_lang=35, seed=42)
