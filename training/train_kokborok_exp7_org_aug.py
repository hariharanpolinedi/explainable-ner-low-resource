"""
Experiment 7: Kokborok ORG Augmentation Multi-Seed Training & Gold-Dev Evaluation.

Evaluates 0x, 0.5x, and 1.0x ORG Augmentation across 3 fixed seeds (42, 123, 2026).
Preserves the frozen baseline checkpoint (checkpoints/kokborok_exp4_best.pt).
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
print(f"Using device: {device} ({gpu_name})")

languages = ("konkani", "assamese", "sinhala", "kokborok")
w_id_4, c_id_4 = load_or_create_multilingual_vocabs(PROC_DIR, languages=languages)

dev_conll = GOLD_DIR / "kokborok" / "dev.conll"
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

def train_and_eval_run(pool_name, pool_path, seed, lr_head=5e-5, lr_emb=3e-5, epochs=6, batch_size=32):
    set_global_seed(seed)
    start_time = time.time()
    
    train_ds = MultilingualNERDataset(lang_conll_paths={"kokborok": [pool_path]}, word_to_id=w_id_4, char_to_id=c_id_4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_multilingual)

    model = NERMultilingualModel(
        word_vocab_size=len(w_id_4),
        char_vocab_size=len(c_id_4),
        num_tags=9,
        num_languages=4,
        crf_config="per_language",
        hidden_dim=128,
    ).to(device)

    # Warm-start from kokborok_exp4_best.pt (frozen baseline)
    init_path = CHECKPOINTS_DIR / "kokborok_exp4_best.pt"
    sd = torch.load(init_path, map_location=device)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)

    # Freeze shared encoder, attention, and non-Kokborok heads
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.attention.parameters():
        p.requires_grad = False
    for p in model.lang_embedding.parameters():
        p.requires_grad = False
    for i in range(3):
        for p in model.fc_heads[i].parameters():
            p.requires_grad = False
        for p in model.crf_heads[i].parameters():
            p.requires_grad = False

    # Trainable: Kokborok Head 3 + Word/Char Embeddings
    for p in model.fc_heads[3].parameters():
        p.requires_grad = True
    for p in model.crf_heads[3].parameters():
        p.requires_grad = True
    for p in model.word_embedding.parameters():
        p.requires_grad = True
    for p in model.char_cnn.parameters():
        p.requires_grad = True

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
        p = d_ov["entity_precision"] * 100
        r = d_ov["entity_recall"] * 100
        tp = d_ov["tp"]
        fp = d_ov["fp"]
        fn = d_ov["fn"]
        
        if f1 >= best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dm
            best_epoch = epoch

    elapsed_time = time.time() - start_time
    return {
        'pool': pool_name,
        'seed': seed,
        'best_epoch': best_epoch,
        'epochs': epochs,
        'lr_head': lr_head,
        'lr_emb': lr_emb,
        'batch_size': batch_size,
        'optimizer': 'AdamW',
        'gpu': gpu_name,
        'elapsed_time_sec': elapsed_time,
        'best_f1': best_f1,
        'metrics': best_metrics,
        'best_state': best_state
    }

def main():
    pools = [
        ("0x", PROC_DIR / "kokborok_exp7_train_0x.conll"),
        ("0.5x", PROC_DIR / "kokborok_exp7_train_05x.conll"),
        ("1.0x", PROC_DIR / "kokborok_exp7_train_1x.conll"),
    ]
    seeds = [42, 123, 2026]
    
    all_results = {}
    
    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT 7: KOKBOROK ORG AUGMENTATION MULTI-SEED RUNS")
    print("=" * 80)
    
    for pool_name, pool_path in pools:
        all_results[pool_name] = []
        print(f"\n--- Testing Configuration: Augmentation Ratio = {pool_name} ---")
        for seed in seeds:
            res = train_and_eval_run(pool_name, pool_path, seed, epochs=6)
            ov = res['metrics']['overall']
            by_type = res['metrics']['by_type']
            
            # Save checkpoint for this run
            ckpt_save_path = CHECKPOINTS_DIR / f"kokborok_exp7_org_aug_{pool_name}_seed{seed}.pt"
            torch.save({
                'model_state_dict': res['best_state'],
                'pool': pool_name,
                'seed': seed,
                'metrics': res['metrics']
            }, ckpt_save_path)
            
            print(f"Ratio: {pool_name:<4} | Seed: {seed:4d} | Best Ep: {res['best_epoch']} | Dev F1: {ov['entity_f1']*100:5.2f}% (P: {ov['entity_precision']*100:5.2f}%, R: {ov['entity_recall']*100:5.2f}%) | TP: {ov['tp']}, FP: {ov['fp']}, FN: {ov['fn']} | PER: {by_type['PER']['f1']*100:5.1f}%, LOC: {by_type['LOC']['f1']*100:5.1f}%, ORG: {by_type['ORG']['f1']*100:5.1f}%, MISC: {by_type['MISC']['f1']*100:5.1f}% | Time: {res['elapsed_time_sec']:.1f}s")
            all_results[pool_name].append(res)

    print("\n" + "=" * 80)
    print("EXPERIMENT 7 MULTI-SEED STATISTICAL SUMMARY (GOLD-DEV)")
    print("=" * 80)
    
    summary_data = {}
    for pool_name in ["0x", "0.5x", "1.0x"]:
        runs = all_results[pool_name]
        f1s = [r['metrics']['overall']['entity_f1'] * 100 for r in runs]
        ps = [r['metrics']['overall']['entity_precision'] * 100 for r in runs]
        rs = [r['metrics']['overall']['entity_recall'] * 100 for r in runs]
        org_f1s = [r['metrics']['by_type']['ORG']['f1'] * 100 for r in runs]
        loc_f1s = [r['metrics']['by_type']['LOC']['f1'] * 100 for r in runs]
        per_f1s = [r['metrics']['by_type']['PER']['f1'] * 100 for r in runs]
        misc_f1s = [r['metrics']['by_type']['MISC']['f1'] * 100 for r in runs]
        
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        min_f1 = np.min(f1s)
        max_f1 = np.max(f1s)
        
        summary_data[pool_name] = {
            'mean_f1': float(mean_f1),
            'std_f1': float(std_f1),
            'min_f1': float(min_f1),
            'max_f1': float(max_f1),
            'mean_p': float(np.mean(ps)),
            'mean_r': float(np.mean(rs)),
            'mean_org_f1': float(np.mean(org_f1s)),
            'mean_loc_f1': float(np.mean(loc_f1s)),
            'mean_per_f1': float(np.mean(per_f1s)),
            'mean_misc_f1': float(np.mean(misc_f1s)),
        }
        
        print(f"Aug Ratio {pool_name:<4} -> Overall Dev F1: {mean_f1:5.2f}% ± {std_f1:4.2f}% (Min: {min_f1:5.2f}%, Max: {max_f1:5.2f}%) | Precision: {np.mean(ps):5.2f}%, Recall: {np.mean(rs):5.2f}% | PER: {np.mean(per_f1s):5.1f}%, LOC: {np.mean(loc_f1s):5.1f}%, ORG: {np.mean(org_f1s):5.1f}%, MISC: {np.mean(misc_f1s):5.1f}%")

    # Save summary json
    with open(PROJECT_ROOT / "reports" / "kokborok_exp7_multiseed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

if __name__ == "__main__":
    main()
