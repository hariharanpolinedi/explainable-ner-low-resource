"""
Experiment 4: Optimal Kokborok-Only F1 Optimization across Multiple Seeds.

Strategy:
1. Warm-start from kokborok_improved_v2.pt (the strongest 87.18% dev baseline).
2. Train Kokborok Head + Embeddings on precision-balanced data (including hard-negatives for Sensor/Social).
3. Evaluate across 3 fixed random seeds (42, 123, 2026) to ensure stability.
4. Track exact TP/FP/FN and Precision/Recall balance.
"""

import copy, json, random, sys, os
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
languages = ("konkani", "assamese", "sinhala", "kokborok")
w_id_4, c_id_4 = load_or_create_multilingual_vocabs(PROC_DIR, languages=languages)

train_conll = PROC_DIR / "kokborok_exp3_precision_train.conll"
dev_conll = GOLD_DIR / "kokborok" / "dev.conll"

train_ds = MultilingualNERDataset(lang_conll_paths={"kokborok": [train_conll]}, word_to_id=w_id_4, char_to_id=c_id_4)
dev_ds = MultilingualNERDataset(lang_conll_paths={"kokborok": [dev_conll]}, word_to_id=w_id_4, char_to_id=c_id_4)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn_multilingual)
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

def run_seed_experiment(seed, lr_head=5e-5, lr_emb=3e-5, epochs=8):
    set_global_seed(seed)
    print(f"\n" + "=" * 60)
    print(f"RUNNING EXPERIMENT 4 | SEED = {seed} | LR_HEAD = {lr_head} | LR_EMB = {lr_emb}")
    print("=" * 60)
    
    model = NERMultilingualModel(
        word_vocab_size=len(w_id_4),
        char_vocab_size=len(c_id_4),
        num_tags=9,
        num_languages=4,
        crf_config="per_language",
        hidden_dim=128,
    ).to(device)

    # Warm-start from kokborok_improved_v2.pt
    v2_path = CHECKPOINTS_DIR / "kokborok_improved_v2.pt"
    sd = torch.load(v2_path, map_location=device)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)

    # Freeze shared encoder and other language heads
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
    print(f"Initial Dev: P={ov['entity_precision']*100:.2f}%, R={ov['entity_recall']*100:.2f}%, F1={ov['entity_f1']*100:.2f}% (TP={ov['tp']}, FP={ov['fp']}, FN={ov['fn']})")

    best_f1 = ov["entity_f1"] * 100
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = init_m

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
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dm
            marker = f"* BEST ({f1:.2f}%)"
        print(f"Ep {epoch:02d}/{epochs:02d} | Loss: {total_loss:.4f} | Dev F1: {f1:5.2f}% (P:{p:5.2f}%, R:{r:5.2f}%) | TP:{tp:2d}, FP:{fp:2d}, FN:{fn:2d} {marker}")

    return best_f1, best_metrics, best_state

seeds = [42, 123, 2026]
results_by_seed = {}

best_overall_f1 = -1.0
best_overall_state = None
best_overall_metrics = None

for s in seeds:
    f1, metrics, state = run_seed_experiment(s, lr_head=5e-5, lr_emb=3e-5, epochs=6)
    results_by_seed[s] = {"f1": f1, "metrics": metrics["overall"]}
    if f1 > best_overall_f1:
        best_overall_f1 = f1
        best_overall_state = state
        best_overall_metrics = metrics

print("\n" + "=" * 70)
print("EXPERIMENT 4 MULTI-SEED SUMMARY (GOLD-DEV)")
print("=" * 70)
f1_list = [v["f1"] for v in results_by_seed.values()]
print(f"Mean Dev F1:   {np.mean(f1_list):.2f}% ± {np.std(f1_list):.2f}%")
print(f"Best Dev F1:   {best_overall_f1:.2f}%")
for s, v in results_by_seed.items():
    ov = v["metrics"]
    print(f"  Seed {s:4d} | F1: {v['f1']:5.2f}% | P: {ov['entity_precision']*100:5.2f}% | R: {ov['entity_recall']*100:5.2f}% | TP: {ov['tp']}, FP: {ov['fp']}, FN: {ov['fn']}")

# Save the best candidate
final_ckpt_path = CHECKPOINTS_DIR / "kokborok_exp4_best.pt"
torch.save({
    "model_state_dict": best_overall_state,
    "dev_metrics": best_overall_metrics,
    "experiment": "Experiment 4: Kokborok F1 Optimal Candidate",
    "multi_seed_results": results_by_seed
}, final_ckpt_path)
print(f"\nSaved optimal candidate to {final_ckpt_path.name}")
