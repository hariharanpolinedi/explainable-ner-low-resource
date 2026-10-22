"""
Experiment 8: Kokborok Contextual LOC-vs-ORG Disambiguation Multi-Seed Training & Evaluation.

Trains only Kokborok-specific parameters across 3 deterministic random seeds (42, 123, 2026).
Evaluates strictly on Gold-Dev against the 94.44% baseline.
Preserves the authoritative frozen benchmark checkpoint (checkpoints/kokborok_exp4_best.pt).
"""

import copy
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("c:/Users/harih/Desktop/NLP-MARK-3")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.seqeval_metrics import compute_seqeval_metrics
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import (
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from training.train_charcnn import ID_TO_TAG
from utils.seed import set_global_seed

PROC_DIR = PROJECT_ROOT / "data" / "processed"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

languages = ("konkani", "assamese", "sinhala", "kokborok")
w_id_4, c_id_4 = load_or_create_multilingual_vocabs(PROC_DIR, languages=languages)

train_conll = PROC_DIR / "kokborok_exp8_context_train.conll"
dev_conll = GOLD_DIR / "kokborok" / "dev.conll"

train_ds = MultilingualNERDataset(lang_conll_paths={"kokborok": [train_conll]}, word_to_id=w_id_4, char_to_id=c_id_4)
dev_ds = MultilingualNERDataset(lang_conll_paths={"kokborok": [dev_conll]}, word_to_id=w_id_4, char_to_id=c_id_4)

dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)

def evaluate_model(m):
    m.eval()
    all_gold, all_pred = [], []
    with torch.no_grad():
        for batch in dev_loader:
            paths = m(batch["word_ids"].to(device), batch["char_ids"].to(device),
                      lang_ids=batch["lang_ids"].to(device), mask=batch["mask"].to(device))
            for i, slen in enumerate(batch["seq_lengths"]):
                all_gold.append([ID_TO_TAG[tid.item()] for tid in batch["tags"][i, :slen]])
                all_pred.append([ID_TO_TAG[tid] for tid in paths[i][:slen]])
    return compute_seqeval_metrics(all_gold, all_pred)

def run_seed_experiment(seed, lr_head=5e-5, lr_emb=3e-5, epochs=6, batch_size=32):
    set_global_seed(seed)
    start_time = time.time()
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_multilingual)

    model = NERMultilingualModel(
        word_vocab_size=len(w_id_4),
        char_vocab_size=len(c_id_4),
        num_tags=9,
        num_languages=4,
        crf_config="per_language",
        hidden_dim=128,
    ).to(device)

    # Warm-start from authoritative exp4 checkpoint
    init_path = CHECKPOINTS_DIR / "kokborok_exp4_best.pt"
    sd = torch.load(init_path, map_location=device)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)

    # Freeze shared encoder and non-Kokborok heads
    for p in model.encoder.parameters(): p.requires_grad = False
    for p in model.attention.parameters(): p.requires_grad = False
    for p in model.lang_embedding.parameters(): p.requires_grad = False
    for i in range(3):
        for p in model.fc_heads[i].parameters(): p.requires_grad = False
        for p in model.crf_heads[i].parameters(): p.requires_grad = False

    # Trainable: Kokborok Head 3 + Word/Char Embeddings
    for p in model.fc_heads[3].parameters(): p.requires_grad = True
    for p in model.crf_heads[3].parameters(): p.requires_grad = True
    for p in model.word_embedding.parameters(): p.requires_grad = True
    for p in model.char_cnn.parameters(): p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    optimizer = torch.optim.AdamW([
        {"params": model.word_embedding.parameters(), "lr": lr_emb},
        {"params": model.char_cnn.parameters(), "lr": lr_emb},
        {"params": model.fc_heads[3].parameters(), "lr": lr_head},
        {"params": model.crf_heads[3].parameters(), "lr": 2e-5},
    ], weight_decay=1e-4)

    init_m = evaluate_model(model)
    ov = init_m["overall"]
    
    best_f1 = ov["entity_f1"] * 100
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = init_m
    best_epoch = 0

    print(f"\n--- SEED {seed} (Trainable Params: {trainable_params:,} | Frozen: {frozen_params:,}) ---")
    print(f"  Init Ep 00 | Dev F1: {best_f1:5.2f}% (P: {ov['entity_precision']*100:5.2f}%, R: {ov['entity_recall']*100:5.2f}%) | TP: {ov['tp']}, FP: {ov['fp']}, FN: {ov['fn']}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(batch["word_ids"].to(device), batch["char_ids"].to(device),
                         lang_ids=batch["lang_ids"].to(device), tags=batch["tags"].to(device), mask=batch["mask"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            total_loss += loss.item()

        dm = evaluate_model(model)
        d_ov = dm["overall"]
        f1 = d_ov["entity_f1"] * 100
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dm
            best_epoch = epoch
            marker = f"* BEST ({f1:.2f}%)"
        
        print(f"  Ep {epoch:02d}/{epochs:02d}   | Loss: {total_loss:7.2f} | Dev F1: {f1:5.2f}% (P: {d_ov['entity_precision']*100:5.2f}%, R: {d_ov['entity_recall']*100:5.2f}%) | TP: {d_ov['tp']:2d}, FP: {d_ov['fp']:2d}, FN: {d_ov['fn']:2d} {marker}")

    elapsed = time.time() - start_time
    return {
        'seed': seed,
        'best_epoch': best_epoch,
        'epochs': epochs,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'gpu': gpu_name,
        'device': str(device),
        'elapsed_sec': elapsed,
        'best_f1': best_f1,
        'metrics': best_metrics,
        'best_state': best_state
    }

def main():
    seeds = [42, 123, 2026]
    results = []

    print("=" * 80)
    print(f"EXPERIMENT 8: CONTEXTUAL DISAMBIGUATION MULTI-SEED RUNS ({gpu_name})")
    print(f"Training Pool: {len(train_ds)} sentences | Gold-Dev: {len(dev_ds)} sentences")
    print("=" * 80)

    for seed in seeds:
        res = run_seed_experiment(seed)
        results.append(res)
        
        # Save separate checkpoint for this seed
        ckpt_path = CHECKPOINTS_DIR / f"kokborok_exp8_context_seed{seed}.pt"
        torch.save({
            'model_state_dict': res['best_state'],
            'seed': seed,
            'metrics': res['metrics']
        }, ckpt_path)
        print(f"  Saved seed checkpoint: {ckpt_path.relative_to(PROJECT_ROOT)}")

    # Summary calculations
    f1s = [r['metrics']['overall']['entity_f1'] * 100 for r in results]
    ps = [r['metrics']['overall']['entity_precision'] * 100 for r in results]
    rs = [r['metrics']['overall']['entity_recall'] * 100 for r in results]
    
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 8 MULTI-SEED STATISTICAL SUMMARY (GOLD-DEV)")
    print("=" * 80)
    print(f"Mean Dev F1:       {mean_f1:5.2f}% +/- {std_f1:4.2f}%")
    print(f"Min Dev F1:        {np.min(f1s):5.2f}%")
    print(f"Max Dev F1:        {np.max(f1s):5.2f}%")
    print(f"Mean Precision:    {np.mean(ps):5.2f}%")
    print(f"Mean Recall:       {np.mean(rs):5.2f}%")
    
    for r in results:
        m = r['metrics']
        ov = m['overall']
        bt = m['by_type']
        print(f"  Seed {r['seed']:4d} | Dev F1: {ov['entity_f1']*100:5.2f}% (P: {ov['entity_precision']*100:5.2f}%, R: {ov['entity_recall']*100:5.2f}%) | PER: {bt['PER']['f1']*100:5.1f}%, LOC: {bt['LOC']['f1']*100:5.1f}%, ORG: {bt['ORG']['f1']*100:5.1f}%, MISC: {bt['MISC']['f1']*100:5.1f}%")

    # Save summary json
    summary_data = {
        'mean_f1': float(mean_f1),
        'std_f1': float(std_f1),
        'min_f1': float(np.min(f1s)),
        'max_f1': float(np.max(f1s)),
        'mean_precision': float(np.mean(ps)),
        'mean_recall': float(np.mean(rs)),
        'per_seed': {
            str(r['seed']): {
                'f1': float(r['metrics']['overall']['entity_f1'] * 100),
                'p': float(r['metrics']['overall']['entity_precision'] * 100),
                'r': float(r['metrics']['overall']['entity_recall'] * 100),
                'by_type': {t: float(r['metrics']['by_type'][t]['f1'] * 100) for t in ['PER', 'LOC', 'ORG', 'MISC']}
            } for r in results
        }
    }
    with open(PROJECT_ROOT / "reports" / "kokborok_exp8_multiseed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

if __name__ == "__main__":
    main()
