"""
Phase 14: Cross-Language Interference & Synergy Verification Script.

Executes direct empirical re-verification of the 4 languages under:
1. Pre-integration Reference Baselines (Phase 7 / Cycle 2 Ground Truth)
2. Phase 12 Option (b) Standalone Kokborok (Control)
3. Phase 12 Option (a) Joint 4-Language Warm-Start (Interference Demonstration)
4. Phase 12 Option (c) Staged Comparison Checkpoint
5. Phase 13 Final Checkpoint (checkpoints/final_kokborok.pt)

Strictly evaluates on gold-dev splits (data/gold/{lang}/dev.conll) without touching gold-test.
Produces complete numerical evidence for the Phase 14 synergy report.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.seqeval_metrics import compute_seqeval_metrics
from model.ner_model_v3_attention import NERAttentionModel
from model.ner_model_v4_multilingual import NERMultilingualModel
from preprocessing.kokborok_tokenizer import KokborokTokenizer
from training.multilingual_dataset import (
    LANG_TO_ID,
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from training.train_charcnn import ID_TO_TAG, TAG_TO_ID

GOLD_DIR = PROJECT_ROOT / "data" / "gold"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"


def evaluate_monolingual_dev(
    model: nn.Module,
    dev_path: Path,
    word_to_id: dict[str, int],
    char_to_id: dict[str, int],
    tokenizer: Any,
    device: torch.device,
) -> dict[str, Any]:
    from experiments.kokborok_integration_experiments import (
        MonolingualCoNLLDataset,
        collate_fn_monolingual,
        evaluate_monolingual_model,
    )

    dev_ds = MonolingualCoNLLDataset(
        conll_path=dev_path,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
        tokenizer=tokenizer,
    )
    return evaluate_monolingual_model(model, dev_ds, device)


def evaluate_multilingual_on_dev(
    model: NERMultilingualModel,
    w_id: dict[str, int],
    c_id: dict[str, int],
    device: torch.device,
    languages: tuple[str, ...] = ("konkani", "assamese", "sinhala", "kokborok"),
) -> dict[str, dict[str, Any]]:
    model.eval()
    results: dict[str, dict[str, Any]] = {}

    for lang in languages:
        dev_file = GOLD_DIR / lang / "dev.conll"
        dev_ds = MultilingualNERDataset(
            lang_conll_paths={lang: [dev_file]},
            word_to_id=w_id,
            char_to_id=c_id,
        )
        loader = DataLoader(
            dev_ds,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn_multilingual,
        )

        all_gold: list[list[str]] = []
        all_pred: list[list[str]] = []

        with torch.no_grad():
            for batch in loader:
                word_ids = batch["word_ids"].to(device)
                char_ids = batch["char_ids"].to(device)
                lang_ids = batch["lang_ids"].to(device)
                tags = batch["tags"].to(device)
                mask = batch["mask"].to(device)
                seq_lengths = batch["seq_lengths"]

                paths = model(word_ids, char_ids, lang_ids=lang_ids, mask=mask)
                for i, slen in enumerate(seq_lengths):
                    gold_seq = [ID_TO_TAG[tid.item()] for tid in tags[i, :slen]]
                    pred_seq = [ID_TO_TAG[tid] for tid in paths[i][:slen]]
                    all_gold.append(gold_seq)
                    all_pred.append(pred_seq)

        metrics = compute_seqeval_metrics(all_gold, all_pred)
        results[lang] = {
            "entity_f1": round(metrics["overall"]["entity_f1"] * 100.0, 2),
            "entity_precision": round(metrics["overall"]["entity_precision"] * 100.0, 2),
            "entity_recall": round(metrics["overall"]["entity_recall"] * 100.0, 2),
            "token_accuracy": round(metrics["overall"]["token_accuracy"] * 100.0, 2),
            "by_type": metrics["by_type"],
        }

    return results


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"PHASE 14: CROSS-LANGUAGE RE-VERIFICATION & SYNERGY AUDIT ON {device}")
    if device.type == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    languages_4 = ("konkani", "assamese", "sinhala", "kokborok")
    w_id_4, c_id_4 = load_or_create_multilingual_vocabs(PROC_DIR, languages=languages_4)

    # 1. Evaluate Final Checkpoint: checkpoints/final_kokborok.pt
    print("\n--- Evaluating Phase 13 Final Checkpoint (final_kokborok.pt) ---")
    final_ckpt_path = CHECKPOINTS_DIR / "final_kokborok.pt"
    if not final_ckpt_path.is_file():
        raise FileNotFoundError(f"Missing {final_ckpt_path}")

    model_final = NERMultilingualModel(
        word_vocab_size=len(w_id_4),
        char_vocab_size=len(c_id_4),
        num_tags=9,
        num_languages=4,
        crf_config="per_language",
    ).to(device)

    final_ckpt = torch.load(final_ckpt_path, map_location=device)
    model_final.load_state_dict(final_ckpt["model_state_dict"])
    final_results = evaluate_multilingual_on_dev(model_final, w_id_4, c_id_4, device, languages=languages_4)

    print("Final Model Results on Official Gold Dev:")
    for lang in languages_4:
        r = final_results[lang]
        print(f"  {lang.capitalize():10s}: F1={r['entity_f1']:5.2f}% | P={r['entity_precision']:5.2f}% | R={r['entity_recall']:5.2f}% | Acc={r['token_accuracy']:5.2f}%")

    # 2. Check Phase 12 Option (c) Comparison Checkpoint
    print("\n--- Evaluating Phase 12 Option (c) Checkpoint (staged_optionC_kokborok.pt) ---")
    staged_c_path = CHECKPOINTS_DIR / "staged_optionC_kokborok.pt"
    staged_c_results = {}
    if staged_c_path.is_file():
        ckpt_c = torch.load(staged_c_path, map_location=device)
        model_final.load_state_dict(ckpt_c["model_state_dict"])
        staged_c_results = evaluate_multilingual_on_dev(model_final, w_id_4, c_id_4, device, languages=languages_4)
        for lang in languages_4:
            r = staged_c_results[lang]
            print(f"  {lang.capitalize():10s}: F1={r['entity_f1']:5.2f}% | P={r['entity_precision']:5.2f}% | R={r['entity_recall']:5.2f}% | Acc={r['token_accuracy']:5.2f}%")

    # 3. Check Phase 12 Option (a) Joint Warm-Start Checkpoint
    print("\n--- Evaluating Phase 12 Option (a) Checkpoint (joint4lang_optionA_kokborok.pt) ---")
    joint_a_path = CHECKPOINTS_DIR / "joint4lang_optionA_kokborok.pt"
    joint_a_results = {}
    if joint_a_path.is_file():
        ckpt_a = torch.load(joint_a_path, map_location=device)
        model_final.load_state_dict(ckpt_a["model_state_dict"])
        joint_a_results = evaluate_multilingual_on_dev(model_final, w_id_4, c_id_4, device, languages=languages_4)
        for lang in languages_4:
            r = joint_a_results[lang]
            print(f"  {lang.capitalize():10s}: F1={r['entity_f1']:5.2f}% | P={r['entity_precision']:5.2f}% | R={r['entity_recall']:5.2f}% | Acc={r['token_accuracy']:5.2f}%")

    # 4. Check Phase 12 Option (b) Standalone Checkpoint
    print("\n--- Evaluating Phase 12 Option (b) Standalone Checkpoint (standalone_kokborok_optionB.pt) ---")
    standalone_b_path = CHECKPOINTS_DIR / "standalone_kokborok_optionB.pt"
    standalone_b_results = {}
    if standalone_b_path.is_file():
        ckpt_b = torch.load(standalone_b_path, map_location="cpu")
        standalone_b_results = ckpt_b.get("dev_metrics", {})
        print(f"  Kokborok Standalone: F1={standalone_b_results.get('entity_f1', 0.0):.2f}% | P={standalone_b_results.get('entity_precision', 0.0):.2f}% | R={standalone_b_results.get('entity_recall', 0.0):.2f}%")

    # 5. Compile Master Re-Verification Dictionary
    master_audit = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "pre_integration_references": {
            "konkani": 96.69,
            "assamese": 97.18,
            "sinhala": 96.30,
            "kokborok": 0.00,
        },
        "option_b_standalone": standalone_b_results,
        "option_a_joint": joint_a_results,
        "option_c_comparison": staged_c_results,
        "final_model_phase13": final_results,
    }

    out_json = REPORTS_DIR / "phase14_synergy_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(master_audit, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED AUDIT DATA] {out_json.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
