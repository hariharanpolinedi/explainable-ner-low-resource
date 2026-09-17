"""
Sinhala Raw-Count Audit on Locked Gold-Dev.

Recomputes PER, LOC, ORG, MISC, and Overall metrics (Precision, Recall, F1,
and raw TP, FP, FN, Support = TP+FN, Predicted = TP+FP) directly from gold-dev
predictions of checkpoints/conjunct_fixed_sinhala.pt and checkpoints/augmented_sinhala.pt.

Explicitly flags any entity class with fewer than 10 instances (project reliability threshold).
Strictly read-only; no training or external dependencies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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


def load_model(
    ckpt_path: Path,
    device: torch.device,
    word_to_id: dict[str, int],
    char_to_id: dict[str, int],
) -> NERMultilingualModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model = NERMultilingualModel(
        word_vocab_size=len(word_to_id),
        char_vocab_size=len(char_to_id),
        num_tags=9,
        num_languages=3,
        hidden_dim=128,
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def evaluate_checkpoint_raw(
    ckpt_path: Path,
    dev_path: Path,
    word_to_id: dict[str, int],
    char_to_id: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    model = load_model(ckpt_path, device, word_to_id, char_to_id)
    ds = MultilingualNERDataset(
        lang_conll_paths={"sinhala": [dev_path]},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )
    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn_multilingual,
    )

    all_golds: list[list[str]] = []
    all_preds: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)
            seq_lengths = batch["seq_lengths"]

            paths = model(word_ids, char_ids, lang_ids=lang_ids, mask=mask)

            for i in range(len(seq_lengths)):
                slen = seq_lengths[i]
                g = [ID_TO_TAG[t.item()] for t in tags[i, :slen]]
                p = [ID_TO_TAG[t] for t in paths[i][:slen]]
                all_golds.append(g)
                all_preds.append(p)

    metrics = compute_seqeval_metrics(all_golds, all_preds)
    return {
        "metrics": metrics,
        "all_golds": all_golds,
        "all_preds": all_preds,
    }


def audit_sinhala_raw_counts(threshold: int = 10) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    gold_dev = PROJECT_ROOT / "data" / "gold" / "sinhala" / "dev.conll"
    base_ckpt = PROJECT_ROOT / "checkpoints" / "augmented_sinhala.pt"
    fixed_ckpt = PROJECT_ROOT / "checkpoints" / "conjunct_fixed_sinhala.pt"

    w_to_id, c_to_id = load_or_create_multilingual_vocabs(proc_dir)

    print("=" * 85)
    print("SINHALA RAW-COUNT AUDIT ON LOCKED GOLD-DEV (data/gold/sinhala/dev.conll)")
    print(f"Reliability Threshold: < {threshold} instances is flagged as statistically fragile")
    print("=" * 85)

    base_res = evaluate_checkpoint_raw(base_ckpt, gold_dev, w_to_id, c_to_id, device)
    fixed_res = evaluate_checkpoint_raw(fixed_ckpt, gold_dev, w_to_id, c_to_id, device)

    ckpts = [
        ("Baseline (augmented_sinhala.pt)", base_res["metrics"]),
        ("Retrained (conjunct_fixed_sinhala.pt)", fixed_res["metrics"]),
    ]

    for name, m in ckpts:
        print(f"\n--- {name} ---")
        ov = m["overall"]
        print(
            f"Overall: P={ov['entity_precision']*100:.2f}%, R={ov['entity_recall']*100:.2f}%, "
            f"F1={ov['entity_f1']*100:.2f}%, TokenAcc={ov['token_accuracy']*100:.2f}%"
        )
        print(f"{'Entity':<6} | {'P (%)':<7} | {'R (%)':<7} | {'F1 (%)':<7} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Support':<8} | {'Flag (<10)'}")
        print("-" * 75)

        for etype in ["PER", "LOC", "ORG", "MISC"]:
            sub = m["by_type"][etype]
            p = sub["precision"] * 100.0
            r = sub["recall"] * 100.0
            f1 = sub["f1"] * 100.0
            tp = sub["tp"]
            fp = sub["fp"]
            fn = sub["fn"]
            support = tp + fn
            flag = "⚠️ FRAGILE (<10 instances)" if support < threshold else "ROBUST (>=10)"
            print(f"{etype:<6} | {p:6.2f}% | {r:6.2f}% | {f1:6.2f}% | {tp:4d} | {fp:4d} | {fn:4d} | {support:8d} | {flag}")

    # Summary comparison
    print("\n" + "=" * 85)
    print("RAW COUNT COMPARISON: Baseline vs Retrained (conjunct_fixed)")
    print("=" * 85)
    print(f"{'Entity':<6} | {'Baseline TP/FP/FN (Supp)':<25} | {'Fixed TP/FP/FN (Supp)':<25} | {'Delta F1':<10} | {'Fragility'}")
    print("-" * 85)
    for etype in ["PER", "LOC", "ORG", "MISC"]:
        b_sub = base_res["metrics"]["by_type"][etype]
        f_sub = fixed_res["metrics"]["by_type"][etype]
        b_str = f"TP={b_sub['tp']}, FP={b_sub['fp']}, FN={b_sub['fn']} ({b_sub['tp']+b_sub['fn']})"
        f_str = f"TP={f_sub['tp']}, FP={f_sub['fp']}, FN={f_sub['fn']} ({f_sub['tp']+f_sub['fn']})"
        df1 = (f_sub["f1"] - b_sub["f1"]) * 100.0
        flag = "⚠️ FRAGILE" if (b_sub["tp"] + b_sub["fn"]) < threshold else "ROBUST"
        print(f"{etype:<6} | {b_str:<25} | {f_str:<25} | {df1:+6.2f}pp   | {flag}")


if __name__ == "__main__":
    audit_sinhala_raw_counts()
