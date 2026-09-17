"""
Supplementary Long Multi-Entity Gold-Set Evaluation.

Evaluates pre-patch vs post-patch checkpoints across all three languages on
the newly constructed supplementary gold dataset:
  data/gold/{lang}/supplementary_long_multientity.conll

Metrics measured:
1. Overall Entity Precision, Recall, F1
2. Per-class Precision, Recall, F1 (PER, LOC, ORG, MISC)
3. Explicit resolution of middle-list entities:
   - Identifies multi-entity lists (>=3 entities clustered with delimiters).
   - Identifies middle entities (non-terminal members of list).
   - Evaluates pre vs post detection of middle entities (resolved vs dropped).
4. Generates reports/audit_supplementary_set_results.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from evaluation.seqeval_metrics import compute_seqeval_metrics, extract_entities
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import (
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from training.train_charcnn import ID_TO_TAG


CHECKPOINTS = {
    "konkani": {
        "pre": PROJECT_ROOT / "checkpoints" / "selftrained_konkani_best.pt",
        "post": PROJECT_ROOT / "checkpoints" / "patched_konkani.pt",
    },
    "assamese": {
        "pre": PROJECT_ROOT / "checkpoints" / "augmented_assamese.pt",
        "post": PROJECT_ROOT / "checkpoints" / "patched_assamese.pt",
    },
    "sinhala": {
        "pre": PROJECT_ROOT / "checkpoints" / "augmented_sinhala.pt",
        "post": PROJECT_ROOT / "checkpoints" / "patched_sinhala.pt",
    },
}

CONJUNCTIONS = {
    "konkani": {"आनी", "आणि", "वा", "तशेंच", ","},
    "assamese": {"আৰু", "বা", "তেনেদৰে", "লগতে", ","},
    "sinhala": {"සහ", "හා", "හෝ", "මෙන්ම", ","},
}


def find_middle_entities(
    tokens: list[str],
    gold_entities: list[tuple[str, int, int]],
    lang: str,
) -> list[tuple[str, int, int]]:
    """
    Identify middle entities in comma/conjunction-separated lists of >=3 entities.
    An entity is 'middle' if it occurs strictly between two other entities in a list cluster.
    """
    if len(gold_entities) < 3:
        return []

    # Sort entities by start index
    sorted_ents = sorted(gold_entities, key=lambda x: x[1])
    conj_set = CONJUNCTIONS.get(lang, {","})

    # Cluster entities that appear close to each other (intervening tokens <= 5)
    clusters: list[list[tuple[str, int, int]]] = []
    current_cluster = [sorted_ents[0]]

    for i in range(1, len(sorted_ents)):
        prev_end = sorted_ents[i - 1][2]
        curr_start = sorted_ents[i][1]
        intervening = tokens[prev_end:curr_start]

        # Check if intervening tokens are few and contain punctuation or conjunction
        has_delim = any(t in conj_set or "," in t for t in intervening) or len(intervening) <= 3
        if (curr_start - prev_end) <= 6 and has_delim:
            current_cluster.append(sorted_ents[i])
        else:
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
            current_cluster = [sorted_ents[i]]

    if len(current_cluster) >= 3:
        clusters.append(current_cluster)

    middle_ents: list[tuple[str, int, int]] = []
    for cluster in clusters:
        # Middle entities are those not at the extreme ends of the cluster
        for ent in cluster[1:-1]:
            middle_ents.append(ent)

    return middle_ents


def evaluate_model_on_dataset(
    model: NERMultilingualModel,
    dataloader: DataLoader,
    device: torch.device,
    lang: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run evaluation and collect predictions and entity extractions."""
    model.eval()
    all_golds: list[list[str]] = []
    all_preds: list[list[str]] = []
    all_tokens: list[list[str]] = []

    with torch.no_grad():
        for b in dataloader:
            paths = model(
                b["word_ids"].to(device),
                b["char_ids"].to(device),
                lang_ids=b["lang_ids"].to(device),
                mask=b["mask"].to(device),
            )
            for i in range(len(b["seq_lengths"])):
                slen = b["seq_lengths"][i]
                g_tags = [ID_TO_TAG[t.item()] for t in b["tags"][i, :slen]]
                p_tags = [ID_TO_TAG[t] for t in paths[i][:slen]]
                toks = b["tokens"][i][:slen]

                all_golds.append(g_tags)
                all_preds.append(p_tags)
                all_tokens.append(toks)

    metrics = compute_seqeval_metrics(all_golds, all_preds)

    sentence_records = []
    for g_tags, p_tags, toks in zip(all_golds, all_preds, all_tokens):
        g_ents = list(extract_entities(g_tags))
        p_ents = list(extract_entities(p_tags))
        mid_ents = find_middle_entities(toks, g_ents, lang)

        sentence_records.append({
            "tokens": toks,
            "gold_tags": g_tags,
            "pred_tags": p_tags,
            "gold_entities": g_ents,
            "pred_entities": p_ents,
            "middle_entities": mid_ents,
        })

    return metrics, sentence_records


def run_supplementary_evaluation() -> dict[str, Any]:
    print("=" * 80)
    print("RE-MEASUREMENT ON SUPPLEMENTARY LONG MULTI-ENTITY GOLD DATASET")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    proc_dir = PROJECT_ROOT / "data" / "processed"
    w_to_id, c_to_id = load_or_create_multilingual_vocabs(proc_dir)

    results: dict[str, Any] = {}

    for lang in ["konkani", "assamese", "sinhala"]:
        print(f"\n--- Evaluating [{lang.upper()}] ---")
        supp_path = PROJECT_ROOT / "data" / "gold" / lang / "supplementary_long_multientity.conll"
        if not supp_path.is_file():
            print(f"Error: supplementary set not found at {supp_path}")
            continue

        ds = MultilingualNERDataset({lang: [supp_path]}, w_to_id, c_to_id)
        dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn_multilingual)

        # Load Pre-Patch Model
        pre_ckpt_path = CHECKPOINTS[lang]["pre"]
        pre_sd = torch.load(pre_ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in pre_sd:
            pre_sd = pre_sd["model_state_dict"]
        model_pre = NERMultilingualModel(len(w_to_id), len(c_to_id), 9, 3, hidden_dim=128).to(device)
        model_pre.load_state_dict(pre_sd, strict=True)

        pre_metrics, pre_records = evaluate_model_on_dataset(model_pre, dl, device, lang)

        # Load Post-Patch Model
        post_ckpt_path = CHECKPOINTS[lang]["post"]
        post_sd = torch.load(post_ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in post_sd:
            post_sd = post_sd["model_state_dict"]
        model_post = NERMultilingualModel(len(w_to_id), len(c_to_id), 9, 3, hidden_dim=128).to(device)
        model_post.load_state_dict(post_sd, strict=True)

        post_metrics, post_records = evaluate_model_on_dataset(model_post, dl, device, lang)

        # Middle entity analysis
        total_middle_entities = 0
        pre_middle_tp = 0
        post_middle_tp = 0

        for r_pre, r_post in zip(pre_records, post_records):
            mid_ents = r_pre["middle_entities"]
            total_middle_entities += len(mid_ents)

            pre_preds_set = set(r_pre["pred_entities"])
            post_preds_set = set(r_post["pred_entities"])

            for m in mid_ents:
                if m in pre_preds_set:
                    pre_middle_tp += 1
                if m in post_preds_set:
                    post_middle_tp += 1

        pre_middle_fn = total_middle_entities - pre_middle_tp
        post_middle_fn = total_middle_entities - post_middle_tp
        pre_middle_recall = (pre_middle_tp / total_middle_entities * 100.0) if total_middle_entities > 0 else 100.0
        post_middle_recall = (post_middle_tp / total_middle_entities * 100.0) if total_middle_entities > 0 else 100.0

        pre_f1 = pre_metrics["overall"]["entity_f1"] * 100.0
        post_f1 = post_metrics["overall"]["entity_f1"] * 100.0

        lang_res = {
            "sentence_count": len(pre_records),
            "pre_patch": {
                "overall": {
                    "precision": round(pre_metrics["overall"]["entity_precision"] * 100.0, 2),
                    "recall": round(pre_metrics["overall"]["entity_recall"] * 100.0, 2),
                    "f1": round(pre_f1, 2),
                },
                "by_type": {
                    t: {
                        "precision": round(m["precision"] * 100.0, 2),
                        "recall": round(m["recall"] * 100.0, 2),
                        "f1": round(m["f1"] * 100.0, 2),
                        "tp": m["tp"],
                        "fp": m["fp"],
                        "fn": m["fn"],
                    }
                    for t, m in pre_metrics["by_type"].items()
                },
                "middle_tp": pre_middle_tp,
                "middle_fn": pre_middle_fn,
                "middle_recall": round(pre_middle_recall, 2),
            },
            "post_patch": {
                "overall": {
                    "precision": round(post_metrics["overall"]["entity_precision"] * 100.0, 2),
                    "recall": round(post_metrics["overall"]["entity_recall"] * 100.0, 2),
                    "f1": round(post_f1, 2),
                },
                "by_type": {
                    t: {
                        "precision": round(m["precision"] * 100.0, 2),
                        "recall": round(m["recall"] * 100.0, 2),
                        "f1": round(m["f1"] * 100.0, 2),
                        "tp": m["tp"],
                        "fp": m["fp"],
                        "fn": m["fn"],
                    }
                    for t, m in post_metrics["by_type"].items()
                },
                "middle_tp": post_middle_tp,
                "middle_fn": post_middle_fn,
                "middle_recall": round(post_middle_recall, 2),
            },
            "middle_entities_total": total_middle_entities,
            "delta_f1": round(post_f1 - pre_f1, 2),
            "delta_middle_recall": round(post_middle_recall - pre_middle_recall, 2),
        }
        results[lang] = lang_res

        print(f"[{lang.upper()}] Sentences: {len(pre_records)} | Total Middle Ents: {total_middle_entities}")
        print(f"  Pre-Patch  Overall F1: {pre_f1:.2f}% | Middle Recall: {pre_middle_recall:.2f}% ({pre_middle_tp}/{total_middle_entities})")
        print(f"  Post-Patch Overall F1: {post_f1:.2f}% | Middle Recall: {post_middle_recall:.2f}% ({post_middle_tp}/{total_middle_entities})")
        print(f"  Delta F1: {lang_res['delta_f1']:+.2f}% | Delta Middle Recall: {lang_res['delta_middle_recall']:+.2f}%")

    # Generate Markdown Report
    generate_supplementary_report(results)
    return results


def generate_supplementary_report(results: dict[str, Any]) -> None:
    rep_path = PROJECT_ROOT / "reports" / "audit_supplementary_set_results.md"
    rep_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Supplementary Long Multi-Entity Gold-Set Evaluation Report",
        "",
        "## Executive Summary",
        "",
        "This audit evaluated pre-patch vs post-patch checkpoints on a newly constructed, strictly independent supplementary gold dataset (`data/gold/{lang}/supplementary_long_multientity.conll`).",
        "Unlike the original gold-dev set (which contained 0/0 missed middle entities and virtually zero multi-entity lists in long sentences), this dataset was specifically curated to test long sentences (>= 30 tokens) containing multi-entity comma/conjunction lists.",
        "",
        "## Evaluation Results Summary",
        "",
        "| Language | Sentences | Gold Middle Ents | Pre-Patch F1 | Post-Patch F1 | Delta F1 | Pre Middle Recall | Post Middle Recall | Delta Middle Recall |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for lang, r in results.items():
        pre_f1 = r["pre_patch"]["overall"]["f1"]
        post_f1 = r["post_patch"]["overall"]["f1"]
        d_f1 = r["delta_f1"]
        mid_total = r["middle_entities_total"]
        pre_m_rec = r["pre_patch"]["middle_recall"]
        post_m_rec = r["post_patch"]["middle_recall"]
        d_m_rec = r["delta_middle_recall"]
        s_cnt = r["sentence_count"]
        lines.append(
            f"| **{lang.capitalize()}** | {s_cnt} | {mid_total} | {pre_f1:.2f}% | {post_f1:.2f}% | **{d_f1:+.2f}%** | {pre_m_rec:.2f}% ({r['pre_patch']['middle_tp']}/{mid_total}) | {post_m_rec:.2f}% ({r['post_patch']['middle_tp']}/{mid_total}) | **{d_m_rec:+.2f}%** |"
        )

    lines.extend([
        "",
        "## Detailed Per-Language Findings",
        "",
    ])

    for lang, r in results.items():
        lines.extend([
            f"### {lang.capitalize()}",
            f"- **Sentence Count**: {r['sentence_count']} (shortfall noted if below 35 target due to exhaustive raw-corpus ceiling)",
            f"- **Total Multi-Entity Middle Targets**: {r['middle_entities_total']}",
            "",
            "#### Pre-Patch vs Post-Patch Per-Class Breakdown",
            "",
            "| Class | Pre Precision | Pre Recall | Pre F1 | Post Precision | Post Recall | Post F1 | Delta F1 |",
            "|---|---|---|---|---|---|---|---|",
        ])
        classes = sorted(list(set(r["pre_patch"]["by_type"].keys()) | set(r["post_patch"]["by_type"].keys())))
        for c in classes:
            p_pre = r["pre_patch"]["by_type"].get(c, {"precision": 0, "recall": 0, "f1": 0})
            p_post = r["post_patch"]["by_type"].get(c, {"precision": 0, "recall": 0, "f1": 0})
            df = p_post["f1"] - p_pre["f1"]
            lines.append(
                f"| {c} | {p_pre['precision']:.2f}% | {p_pre['recall']:.2f}% | {p_pre['f1']:.2f}% | {p_post['precision']:.2f}% | {p_post['recall']:.2f}% | {p_post['f1']:.2f}% | {df:+.2f}% |"
            )

        lines.extend([
            "",
            f"#### Middle-Entity Resolution Analysis ({lang.capitalize()})",
            f"- **Pre-Patch**: {r['pre_patch']['middle_tp']} / {r['middle_entities_total']} correctly detected ({r['pre_patch']['middle_recall']}%), {r['pre_patch']['middle_fn']} dropped.",
            f"- **Post-Patch**: {r['post_patch']['middle_tp']} / {r['middle_entities_total']} correctly detected ({r['post_patch']['middle_recall']}%), {r['post_patch']['middle_fn']} dropped.",
            f"- **Net Impact**: Delta in middle-entity recall is {r['delta_middle_recall']:+.2f}%.",
            "",
        ])

    lines.extend([
        "## Audit Conclusions from Supplementary Evaluation",
        "",
        "1. **Konkani**: The patched checkpoint represents genuinely retrained weights. On the supplementary long multi-entity set, the empirical delta demonstrates whether the list-augmentation translated to actual generalization on long sentences.",
        "2. **Assamese & Sinhala**: The patched checkpoints are parameter-identical to their pre-patch checkpoints (L2 diff = 0.0000). Consequently, performance on the supplementary set is identically identical (0.00% delta across all metrics) because no weights were updated during patch retraining.",
        "3. **Hypothesis Validation**: The patch's core claim that list-augmentation systematically eliminates middle-entity dropping is unproven for Assamese and Sinhala because retraining did not update their checkpoints, and directly measured on Konkani via this supplementary set.",
        "",
    ])

    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport written to: {rep_path}")


if __name__ == "__main__":
    run_supplementary_evaluation()
