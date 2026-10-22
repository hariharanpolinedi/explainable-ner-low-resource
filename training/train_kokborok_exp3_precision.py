import copy, json, random, sys, os
from pathlib import Path

PROJECT_ROOT = Path("c:/Users/harih/Desktop/NLP-MARK-3")
sys.path.insert(0, str(PROJECT_ROOT))

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
print(f"Training Kokborok Experiment 3 (Precision Recovery) on device: {device}")
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

# Load baseline weights from updated_kokborok_optionC.pt
base_ckpt = CHECKPOINTS_DIR / "updated_kokborok_optionC.pt"
sd = torch.load(base_ckpt, map_location=device)
model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)

# 100% FREEZE shared backbone (BiLSTM, Attention, Lang Embedding, Heads 0-2)
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

# TRAINABLE: Kokborok Head 3 + Word/Char Embeddings
for p in model.fc_heads[3].parameters():
    p.requires_grad = True
for p in model.crf_heads[3].parameters():
    p.requires_grad = True
for p in model.word_embedding.parameters():
    p.requires_grad = True
for p in model.char_cnn.parameters():
    p.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Params: {total_params:,} | Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
print("Verified: BiLSTM, Attention, and Heads 0-2 are 100% FROZEN.")

train_conll = PROC_DIR / "kokborok_exp3_precision_train.conll"
dev_conll = GOLD_DIR / "kokborok" / "dev.conll"

train_ds = MultilingualNERDataset(lang_conll_paths={"kokborok": [train_conll]}, word_to_id=w_id_4, char_to_id=c_id_4)
dev_ds = MultilingualNERDataset(lang_conll_paths={"kokborok": [dev_conll]}, word_to_id=w_id_4, char_to_id=c_id_4)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn_multilingual)
dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)

optimizer = torch.optim.AdamW([
    {"params": model.word_embedding.parameters(), "lr": 1e-4},
    {"params": model.char_cnn.parameters(), "lr": 1e-4},
    {"params": model.fc_heads[3].parameters(), "lr": 2e-4},
    {"params": model.crf_heads[3].parameters(), "lr": 5e-5},
], weight_decay=1e-4)

def evaluate_dev(m):
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

init_m = evaluate_dev(model)
ov = init_m["overall"]
print(f"Initial Dev F1: {ov['entity_f1']*100:.2f}% (P: {ov['entity_precision']*100:.2f}%, R: {ov['entity_recall']*100:.2f}%) | TP: {ov['tp']}, FP: {ov['fp']}, FN: {ov['fn']}")

best_f1 = ov["entity_f1"] * 100
best_state = copy.deepcopy(model.state_dict())
best_metrics = init_m

num_epochs = 10
print(f"\nStarting training for Experiment 3 ({num_epochs} epochs)...")

for epoch in range(1, num_epochs + 1):
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

    dm = evaluate_dev(model)
    d_ov = dm["overall"]
    f1 = d_ov["entity_f1"] * 100
    p = d_ov["entity_precision"] * 100
    r = d_ov["entity_recall"] * 100
    tp = d_ov["tp"]
    fp = d_ov["fp"]
    fn = d_ov["fn"]
    
    status = ""
    if f1 > best_f1:
        best_f1 = f1
        best_state = copy.deepcopy(model.state_dict())
        best_metrics = dm
        status = f"* NEW BEST F1 ({f1:.2f}%)"
    
    print(f"Epoch {epoch:02d}/{num_epochs:02d} | Loss: {total_loss:.4f} | Dev F1: {f1:5.2f}% (P: {p:5.2f}%, R: {r:5.2f}%) | TP: {tp:2d}, FP: {fp:2d}, FN: {fn:2d} {status}")

# Save candidate
out_ckpt = CHECKPOINTS_DIR / "kokborok_exp3_precision_recovered.pt"
torch.save({
    "model_state_dict": best_state,
    "dev_metrics": best_metrics,
    "experiment": "Experiment 3: Hard-Negative Precision Recovery",
    "config": {
        "frozen": "encoder, attention, lang_embedding, heads 0..2",
        "trainable": "word_embedding, char_cnn, fc_heads[3], crf_heads[3]",
        "dataset": "kokborok_exp3_precision_train.conll"
    }
}, out_ckpt)
print(f"\nSaved best model to {out_ckpt.name} with Dev F1: {best_f1:.2f}%")

# Step 8: Multi-language Regression Check
print("\n" + "=" * 80)
print("STEP 8: MULTI-LANGUAGE REGRESSION VERIFICATION")
print("=" * 80)
model.load_state_dict(best_state)
model.eval()

PROTECTED_BASELINES = {"konkani": 96.69, "assamese": 97.18, "sinhala": 96.30}
for lang in ["konkani", "assamese", "sinhala"]:
    lp = GOLD_DIR / lang / "dev.conll"
    lds = MultilingualNERDataset(lang_conll_paths={lang: [lp]}, word_to_id=w_id_4, char_to_id=c_id_4)
    lloader = DataLoader(lds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)
    lg_gold, lg_pred = [], []
    with torch.no_grad():
        for batch in lloader:
            paths = model(batch["word_ids"].to(device), batch["char_ids"].to(device),
                          lang_ids=batch["lang_ids"].to(device), mask=batch["mask"].to(device))
            for i, slen in enumerate(batch["seq_lengths"]):
                lg_gold.append([ID_TO_TAG[tid.item()] for tid in batch["tags"][i, :slen]])
                lg_pred.append([ID_TO_TAG[tid] for tid in paths[i][:slen]])
    lm = compute_seqeval_metrics(lg_gold, lg_pred)
    cur = lm["overall"]["entity_f1"] * 100
    base = PROTECTED_BASELINES[lang]
    delta = cur - base
    status = "[PASS] ZERO REGRESSION" if delta >= -0.10 else "[FAIL] REGRESSION"
    print(f"{lang.upper():<10} | Protected: {base:.2f}% | Current: {cur:.2f}% | Delta: {delta:+.2f} pp | {status}")
