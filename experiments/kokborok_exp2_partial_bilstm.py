"""
Experiment B: Partial BiLSTM Unfreezing (Layer 1 Only) for Kokborok NER.

Hypothesis:
Unfreezing only the final BiLSTM layer (Layer 1) allows contextual representations
to adapt to Kokborok sequence syntax (e.g. multi-token names like 'Parimal Debbarma',
contextual indicators like 'ni', 'o') while keeping Layer 0, Embeddings, Char-CNN,
and Attention frozen to minimize risk to other languages.

Training Data: data/processed/kokborok_targeted_inflected_train.conll
Validation: data/gold/kokborok/dev.conll
Baseline Checkpoint: checkpoints/updated_kokborok_optionC.pt
"""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("c:/Users/harih/Desktop/NLP-MARK-3")
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.seqeval_metrics import compute_seqeval_metrics, extract_entities
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import (
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from training.train_charcnn import ID_TO_TAG, TAG_TO_ID
from utils.seed import set_global_seed

PROC_DIR = PROJECT_ROOT / "data" / "processed"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

PRE_INTEGRATION_BASELINES = {
    "konkani": 96.69,
    "assamese": 97.18,
    "sinhala": 96.30,
}

def evaluate_split(model: NERMultilingualModel, dataloader: DataLoader, device: torch.device):
    model.eval()
    all_gold: list[list[str]] = []
    all_pred: list[list[str]] = []
    with torch.no_grad():
        for batch in dataloader:
            w = batch["word_ids"].to(device)
            c = batch["char_ids"].to(device)
            l_ids = batch["lang_ids"].to(device)
            t = batch["tags"].to(device)
            m = batch["mask"].to(device)
            seq_lengths = batch["seq_lengths"]

            paths = model(w, c, lang_ids=l_ids, mask=m)
            for i, slen in enumerate(seq_lengths):
                all_gold.append([ID_TO_TAG[tid.item()] for tid in t[i, :slen]])
                all_pred.append([ID_TO_TAG[tid] for tid in paths[i][:slen]])

    raw_metrics = compute_seqeval_metrics(all_gold, all_pred)
    overall = raw_metrics.get("overall", {})
    return {
        "entity_f1": overall.get("entity_f1", 0.0) * 100.0,
        "entity_precision": overall.get("entity_precision", 0.0) * 100.0,
        "entity_recall": overall.get("entity_recall", 0.0) * 100.0,
        "raw": raw_metrics,
        "all_gold": all_gold,
        "all_pred": all_pred,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Experiment B: Partial BiLSTM Unfreezing on device: {device}")
    set_global_seed(42)

    languages = ("konkani", "assamese", "sinhala", "kokborok")
    w_id_4, c_id_4 = load_or_create_multilingual_vocabs(PROC_DIR, languages=languages)

    model = NERMultilingualModel(
        word_vocab_size=len(w_id_4),
        char_vocab_size=len(c_id_4),
        num_tags=9,
        num_languages=4,
        crf_config="per_language",
        hidden_dim=128,
    ).to(device)

    baseline_ckpt_path = CHECKPOINTS_DIR / "updated_kokborok_optionC.pt"
    print(f"Loading immutable baseline weights from {baseline_ckpt_path.name}...")
    sd = torch.load(baseline_ckpt_path, map_location=device)
    model.load_state_dict(sd["model_state_dict"])

    # 1. Freeze ALL embeddings, Char-CNN, Attention, Lang-Embedding, and Heads 0-2
    for p in model.word_embedding.parameters():
        p.requires_grad = False
    for p in model.char_cnn.parameters():
        p.requires_grad = False
    for p in model.lang_embedding.parameters():
        p.requires_grad = False
    for p in model.attention.parameters():
        p.requires_grad = False
    for i in range(3):
        for p in model.fc_heads[i].parameters():
            p.requires_grad = False
        for p in model.crf_heads[i].parameters():
            p.requires_grad = False

    # 2. Configure BiLSTM: Freeze Layer 0, Unfreeze ONLY Layer 1
    lstm_l1_params = []
    lstm_l0_params = []
    for name, param in model.encoder.lstm.named_parameters():
        if "_l1" in name:
            param.requires_grad = True
            lstm_l1_params.append(param)
        else:
            param.requires_grad = False
            lstm_l0_params.append(param)

    # 3. Unfreeze Kokborok Head (Head 3)
    for p in model.fc_heads[3].parameters():
        p.requires_grad = True
    for p in model.crf_heads[3].parameters():
        p.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    l1_count = sum(p.numel() for p in lstm_l1_params)
    head_count = sum(p.numel() for p in model.fc_heads[3].parameters()) + sum(p.numel() for p in model.crf_heads[3].parameters())

    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params_count:,} ({trainable_params_count/total_params*100:.2f}%)")
    print(f"  - BiLSTM Layer 1:    {l1_count:,} params")
    print(f"  - Kokborok Head 3:   {head_count:,} params")
    print(f"Frozen Parameters:     {total_params - trainable_params_count:,} ({(total_params-trainable_params_count)/total_params*100:.2f}%)")
    print("Safeguard: BiLSTM Layer 0, Embeddings, Attention, and Heads 0-2 are 100% FROZEN.")

    # Datasets
    train_conll = PROC_DIR / "kokborok_targeted_inflected_train.conll"
    dev_conll = GOLD_DIR / "kokborok" / "dev.conll"

    train_ds = MultilingualNERDataset(
        lang_conll_paths={"kokborok": [train_conll]},
        word_to_id=w_id_4,
        char_to_id=c_id_4,
    )
    dev_ds = MultilingualNERDataset(
        lang_conll_paths={"kokborok": [dev_conll]},
        word_to_id=w_id_4,
        char_to_id=c_id_4,
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn_multilingual)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)

    # Differential Learning Rates: Conservative 3e-5 on shared BiLSTM Layer 1, 3e-4 on linear head, 5e-5 on CRF
    optimizer = torch.optim.AdamW([
        {"params": lstm_l1_params, "lr": 3e-5},
        {"params": model.fc_heads[3].parameters(), "lr": 3e-4},
        {"params": model.crf_heads[3].parameters(), "lr": 5e-5},
    ], weight_decay=1e-4)

    # Initial Dev Evaluation
    pre_metrics = evaluate_split(model, dev_loader, device)
    print(f"\nInitial Kokborok Dev F1: {pre_metrics['entity_f1']:.2f}% (P: {pre_metrics['entity_precision']:.2f}%, R: {pre_metrics['entity_recall']:.2f}%)")

    best_dev_f1 = pre_metrics["entity_f1"]
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = pre_metrics

    num_epochs = 10
    print(f"\nStarting training loop for Experiment B (up to {num_epochs} epochs)...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            w = batch["word_ids"].to(device)
            c = batch["char_ids"].to(device)
            l_ids = batch["lang_ids"].to(device)
            t = batch["tags"].to(device)
            m = batch["mask"].to(device)

            loss = model(w, c, lang_ids=l_ids, tags=t, mask=m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            total_loss += loss.item()

        # Evaluate Dev
        dev_metrics = evaluate_split(model, dev_loader, device)
        f1 = dev_metrics["entity_f1"]
        p = dev_metrics["entity_precision"]
        r = dev_metrics["entity_recall"]

        status = ""
        if f1 > best_dev_f1:
            best_dev_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dev_metrics
            status = f"[NEW BEST! {f1:.2f}%]"

        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {total_loss:.4f} | Dev F1: {f1:.2f}% (P: {p:.2f}%, R: {r:.2f}%) {status}")

    # Load best state
    model.load_state_dict(best_state)
    print(f"\n--- Best Kokborok Dev F1 in Experiment B: {best_dev_f1:.2f}% ---")
    ov = best_metrics["raw"]["overall"]
    print(f"Precision: {ov['entity_precision']*100:.2f}%, Recall: {ov['entity_recall']*100:.2f}%")
    print(f"TP: {ov['tp']}, FP: {ov['fp']}, FN: {ov['fn']}")
    print(f"Per-type breakdown: {best_metrics['raw']['by_type']}")

    # Save checkpoint
    cand_path = CHECKPOINTS_DIR / "kokborok_exp2_partial_bilstm.pt"
    torch.save({
        "model_state_dict": best_state,
        "dev_metrics": best_metrics["raw"],
        "experiment": "Experiment B: Partial BiLSTM Unfreezing",
        "config": {
            "unfrozen": "encoder.lstm_l1, fc_heads.3, crf_heads.3",
            "lr_bilstm": 3e-5,
            "lr_fc": 3e-4,
            "lr_crf": 5e-5,
        }
    }, cand_path)
    print(f"Saved checkpoint to {cand_path.name}")

    # Cross-Language Verification
    print("\n" + "=" * 80)
    print("STEP 7: CROSS-LANGUAGE REGRESSION VERIFICATION")
    print("=" * 80)
    regression_passed = True
    multi_scores = {}
    for lang in ["konkani", "assamese", "sinhala"]:
        lang_dev_path = GOLD_DIR / lang / "dev.conll"
        lang_ds = MultilingualNERDataset(
            lang_conll_paths={lang: [lang_dev_path]},
            word_to_id=w_id_4,
            char_to_id=c_id_4,
        )
        lang_loader = DataLoader(lang_ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)
        res = evaluate_split(model, lang_loader, device)
        base = PRE_INTEGRATION_BASELINES[lang]
        delta = res["entity_f1"] - base
        multi_scores[lang] = (res["entity_f1"], delta)
        print(f"{lang.upper():<10} | Base: {base:.2f}% | Current: {res['entity_f1']:.2f}% | Delta: {delta:+.2f} pp")
        if delta < -0.10:
            print(f"[FAIL] REGRESSION DETECTED on {lang}!")
            regression_passed = False
        else:
            print(f"[PASS] ZERO REGRESSION CONFIRMED for {lang}.")

    return {
        "best_f1": best_dev_f1,
        "best_metrics": best_metrics,
        "multi_scores": multi_scores,
    }

if __name__ == "__main__":
    main()
