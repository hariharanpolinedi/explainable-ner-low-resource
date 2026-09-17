"""
Audit Annotation Independence in long_multientity_eval.conll.

Investigates:
1. The exact computational definition of "Annotator A" and "Annotator B".
2. Why Konkani achieved a perfect Cohen's Kappa = 1.0000.
3. Why Assamese and Sinhala had non-1.0 Kappa.
4. Whether the inter-annotator agreement included or omitted the DedicatedORGRuleTagger tags.
5. Circularity and independence verification across all three languages.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation.agreement_calculator import compute_cohens_kappa
from annotation.annotate_gold import HONORIFICS, load_entity_dictionaries, annotate_sentence
from evaluation.seqeval_metrics import extract_entities
from weak_supervision.org_rule_taggers import DedicatedORGRuleTagger


def parse_eval_file(file_path: Path) -> list[list[tuple[str, str]]]:
    """Parse long_multientity_eval.conll into sentences of (token, adjudicated_tag)."""
    sentences: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split("\t")
                tok = parts[0]
                tag = parts[1] if len(parts) > 1 else "O"
                current.append((tok, tag))
    if current:
        sentences.append(current)
    return sentences


def audit_language_annotation(lang: str) -> dict[str, Any]:
    gaz_dir = PROJECT_ROOT / "data" / "gazetteers"
    entity_dict = load_entity_dictionaries(lang, gaz_dir)
    honorifics = HONORIFICS.get(lang, set())
    org_tagger = DedicatedORGRuleTagger(lang)

    eval_file = PROJECT_ROOT / "data" / "gold" / lang / "long_multientity_eval.conll"
    parsed_sents = parse_eval_file(eval_file)

    # Re-run annotator A, B, and org_tagger on the actual sentences in the file
    overlap_size = min(20, len(parsed_sents))
    disagreements_ab: list[dict[str, Any]] = []

    flat_a: list[str] = []
    flat_b: list[str] = []
    flat_adj: list[str] = []

    rule_tagger_overwrites_in_adj = 0
    total_tokens = 0

    for idx, sent in enumerate(parsed_sents):
        toks = [t[0] for t in sent]
        saved_adj = [t[1] for t in sent]
        total_tokens += len(toks)

        tags_a = annotate_sentence(toks, entity_dict, honorifics, variant="annotator_a")
        tags_b = annotate_sentence(toks, entity_dict, honorifics, variant="annotator_b")
        org_tags = org_tagger.tag_sentence(toks)

        # Trace differences between A and saved adjudicated
        for i in range(len(toks)):
            if tags_a[i] == "O" and saved_adj[i] != "O":
                rule_tagger_overwrites_in_adj += 1

        if idx < overlap_size:
            flat_a.extend(tags_a)
            flat_b.extend(tags_b)
            flat_adj.extend(saved_adj)

            if tags_a != tags_b:
                diffs = [(toks[i], tags_a[i], tags_b[i]) for i in range(len(toks)) if tags_a[i] != tags_b[i]]
                disagreements_ab.append({
                    "sentence_idx": idx,
                    "tokens": " ".join(toks),
                    "differences": diffs,
                })

    # Kappa between Annotator A and Annotator B
    kappa_ab = compute_cohens_kappa(flat_a, flat_b)["kappa"]

    # Kappa between Annotator A and Saved Adjudicated (shows impact of rule tagger)
    kappa_a_vs_adj = compute_cohens_kappa(flat_a, flat_adj)["kappa"]

    # Count entities by type under Annotator A vs Final Adjudicated
    ents_a = []
    ents_adj = []
    for sent in parsed_sents:
        toks = [t[0] for t in sent]
        saved_adj = [t[1] for t in sent]
        tags_a = annotate_sentence(toks, entity_dict, honorifics, variant="annotator_a")
        ents_a.extend(extract_entities(tags_a))
        ents_adj.extend(extract_entities(saved_adj))

    from collections import Counter
    counts_a = Counter(e[0] for e in ents_a)
    counts_adj = Counter(e[0] for e in ents_adj)

    return {
        "language": lang,
        "total_sentences": len(parsed_sents),
        "total_tokens": total_tokens,
        "overlap_sample_size": overlap_size,
        "annotator_a_vs_b_disagreements_count": len(disagreements_ab),
        "annotator_a_vs_b_disagreements": disagreements_ab,
        "kappa_annotator_a_vs_b": round(kappa_ab, 4),
        "kappa_annotator_a_vs_adjudicated": round(kappa_a_vs_adj, 4),
        "rule_tagger_injected_tokens": rule_tagger_overwrites_in_adj,
        "entities_under_gazetteer_only_annotator_a": dict(counts_a),
        "entities_under_final_adjudicated": dict(counts_adj),
    }


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    print("=== ANNOTATION PROCESS AUDIT ===")
    for lang in ["konkani", "assamese", "sinhala"]:
        res = audit_language_annotation(lang)
        print(f"\n--- {lang.upper()} ---")
        print(f"Total sentences: {res['total_sentences']}")
        print(f"Overlap sample size: {res['overlap_sample_size']}")
        print(f"A vs B disagreements in overlap: {res['annotator_a_vs_b_disagreements_count']}")
        print(f"Kappa (Annotator A vs Annotator B): {res['kappa_annotator_a_vs_b']}")
        print(f"Kappa (Annotator A vs Saved Adjudicated): {res['kappa_annotator_a_vs_adjudicated']}")
        print(f"Rule tagger injected tokens into final file: {res['rule_tagger_injected_tokens']}")
        print(f"Entities from Annotator A (gazetteer-only): {res['entities_under_gazetteer_only_annotator_a']}")
        print(f"Entities in Final Adjudicated file: {res['entities_under_final_adjudicated']}")
        if res['annotator_a_vs_b_disagreements']:
            print("Sample disagreements between A and B:")
            for d in res['annotator_a_vs_b_disagreements'][:2]:
                print(f"  Sentence {d['sentence_idx']}: {d['differences']}")
