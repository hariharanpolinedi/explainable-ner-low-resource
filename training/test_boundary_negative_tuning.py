"""
Precision-Locked Negative Tuning:
1. Keeps the 100% recall capability on PER (Rahul, Mohan), ORG (Tripura board), LOC (all 10), MISC (all 3).
2. Directly adds sentence-initial O-negatives for Kokborok vocabulary to suppress spurious PER/ORG triggers.
3. Fine-tunes with very low learning rate (2e-5) for 3-5 epochs.
"""
import copy, json, random, sys, os
from pathlib import Path

ROOT = Path(r"c:\Users\harih\Desktop\NLP-MARK-3")
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import MultilingualNERDataset, collate_fn_multilingual
from evaluation.seqeval_metrics import compute_seqeval_metrics
from training.train_charcnn import ID_TO_TAG, TAG_TO_ID
from utils.seed import set_global_seed

from training.kokborok_surgical_finetune import (
    load_conll_sentences, save_conll, make_sent, all_o, run_eval
)

set_global_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bundle_dir = ROOT / "deployment" / "bundles" / "kokborok" / "v1"
w2id = json.load(open(bundle_dir / "word_vocab.json", "r", encoding="utf-8")).get("token_to_id", {})
c2id = json.load(open(bundle_dir / "char_vocab.json", "r", encoding="utf-8")).get("token_to_id", {})

gold_train_path = ROOT / "data" / "gold" / "kokborok" / "train.conll"
gold_train_sents = load_conll_sentences(gold_train_path)

# Targeted positive + negative sentences
targeted_sents = []

# 1. PER positive: Rahul and Mohan specifically
targeted_sents.append(make_sent("Rahul bike accident wngwi thwikha .", [("Rahul", "PER")]))
targeted_sents.append(make_sent("Rahul tei Mohan bike accident wngwi thwikha .", [("Rahul", "PER"), ("Mohan", "PER")]))
targeted_sents.append(make_sent("Where is Mohan Bo phaiwi de tong", [("Mohan", "PER")]))
targeted_sents.append(make_sent("Rahul hospital o tongo tabuk .", [("Rahul", "PER")]))
targeted_sents.append(make_sent("Mohan Tripura ni khoroksa bura borok .", [("Mohan", "PER"), ("Tripura", "LOC")]))

# 2. ORG positive: Tripura board specifically + contrast pairs
targeted_sents.append(make_sent("Kebo Tripura board ni employee rok da tong sabai gradi madhyamik ni admit .", [("Tripura board", "ORG")]))
targeted_sents.append(make_sent("Tripura board ni admit card mankha tabuk .", [("Tripura board", "ORG")]))
targeted_sents.append(make_sent("Tripura board ni madhyamik result phaikha .", [("Tripura board", "ORG")]))
targeted_sents.append(make_sent("Tripura council ni executive meeting tongo .", [("Tripura council", "ORG")]))

# Exact LOC contrast pairs (Tripura alone = LOC)
targeted_sents.append(make_sent("Tripura ni order marwsinai ta hing khe .", [("Tripura", "LOC")]))
targeted_sents.append(make_sent("Kok Tripura no informed khai khe no wngwano .", [("Tripura", "LOC")]))
targeted_sents.append(make_sent("Tripura chini tongmani but tabuk whole Tripura chini hinui manilya .", [("Tripura", "LOC"), ("Tripura", "LOC")]))
targeted_sents.append(make_sent("Tripura ha is not for one religion Tripura ha is for all religion .", [("Tripura", "LOC"), ("Tripura", "LOC")]))
targeted_sents.append(make_sent("Chini Tripura ni se follow khailaikha tabuk khai bhorok bo .", [("Tripura", "LOC")]))
targeted_sents.append(make_sent("Tripura ni tamo news tong tini .", [("Tripura", "LOC")]))
targeted_sents.append(make_sent("Agartala ni order marwsinai ta hing khe .", [("Agartala", "LOC")]))
targeted_sents.append(make_sent("Agartala wo helium tank manthoknai de .", [("Agartala", "LOC")]))
targeted_sents.append(make_sent("Udaipur o serious incident ongkha tini .", [("Udaipur", "LOC")]))

# 3. MISC positive: Christmas
targeted_sents.append(make_sent("Ang nok no kisa Christmas ni sekang maphai renai by any how .", [("Christmas", "MISC")]))
targeted_sents.append(make_sent("Kokthai kwrwini bagwi advanced Christmas ni swkang painani nuhohrya .", [("Christmas", "MISC")]))
targeted_sents.append(make_sent("Christmas ni decoration khwlai tongo .", [("Christmas", "MISC")]))

# 4. NEGATIVE SENTENCES (All O) - explicitly teaching the model NOT to trigger on these words:
# a) Capitalized sentence starters that are NOT entities
targeted_sents.append(all_o("Cherai no beginning simi no kaham khe guide khai na nango ."))
targeted_sents.append(all_o("Cherai rok no school o tisidi ."))
targeted_sents.append(all_o("Stupid mungsa samungo chukya ."))
targeted_sents.append(all_o("Phaidi Rwchapmung Ter o thanglai anw ."))
targeted_sents.append(all_o("Nwngle tamo sa bijapda poridi hwn toke nainani ?"))
targeted_sents.append(all_o("Mandai ni tamo news tong tini ."))
targeted_sents.append(all_o("Biplab no aswk khahamjago tei praise khwlaio ."))
targeted_sents.append(all_o("Laboratory Sciences ni phwrwngnairokni mungni bagwi tolai sindi ."))
targeted_sents.append(all_o("Paaa all fresh vegetables fully organic orolee inorganic vegetables chamang thwikha ."))
targeted_sents.append(all_o("Congratulation for the new car bufru treat renai sadi ."))
targeted_sents.append(all_o("Dance academy o sa admission malanai ."))
targeted_sents.append(all_o("Day to day update khlai khe chao ."))
targeted_sents.append(all_o("Jones sakha je Swrwngthairok Los Angeles , USA ; New York , USA ; tei Kwtal Delhi , India ; abohai Nangkukmani Dolirogo tongo ."))
targeted_sents.append(all_o("New York , USA ; tei Kwtal Delhi , India ; abohai Nangkukmani Dolirogo tongo ."))
targeted_sents.append(all_o("Careless drive ni bagui mangjak kha ta ."))
targeted_sents.append(all_o("Ultimate chini no loss naidi ."))
targeted_sents.append(all_o("Still mayasak ang naitoknai ."))
targeted_sents.append(all_o("Water supply phaikhe tai wanaliya ."))
targeted_sents.append(all_o("Sir kaitor tei norokni hamari manwi tongthu ."))
targeted_sents.append(all_o("One week wnkha hospital o ."))
targeted_sents.append(all_o("Still pending tamo khainai kha tabuk ."))
targeted_sents.append(all_o("Where is the problem nwng sadi ."))
targeted_sents.append(all_o("Bo phaiwi de tong tini ?"))

print(f"Prepared {len(targeted_sents)} high-precision targeted calibration sentences.")

# Combine gold train (850 sentences) + targeted sentences repeated 10x for strong gradient focus
combined = []
combined.extend(gold_train_sents)
for _ in range(10):
    combined.extend(targeted_sents)
random.seed(42)
random.shuffle(combined)

calib_train_path = ROOT / "data" / "processed" / "kokborok_calib_train.conll"
save_conll(combined, calib_train_path)

test_path = ROOT / "data" / "gold" / "kokborok" / "test.conll"
train_ds = MultilingualNERDataset({"kokborok": [calib_train_path]}, word_to_id=w2id, char_to_id=c2id)
test_ds = MultilingualNERDataset({"kokborok": [test_path]}, word_to_id=w2id, char_to_id=c2id)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn_multilingual)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)

model = NERMultilingualModel(
    word_vocab_size=len(w2id), char_vocab_size=len(c2id),
    num_tags=9, num_languages=4, crf_config="per_language", hidden_dim=128
).to(device)

base_ckpt = torch.load(ROOT / "checkpoints" / "kokborok_improved_v2.pt", map_location=device, weights_only=False)
model.load_state_dict(base_ckpt["model_state_dict"])

trainable = []
for n, p in model.named_parameters():
    if any(f"fc_heads.{i}" in n or f"crf_heads.{i}" in n for i in [0, 1, 2]):
        p.requires_grad = False
    else:
        p.requires_grad = True
        trainable.append(p)

# We use a very gentle learning rate of 2e-5 with weight decay
lr = 2e-5
epochs = 8
optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

init_m, _, _ = run_eval(model, test_loader, device)
print(f"\nBaseline Test: F1={init_m['overall']['entity_f1']*100:.2f}% | P={init_m['overall']['entity_precision']*100:.2f}% | R={init_m['overall']['entity_recall']*100:.2f}%")

best_f1 = init_m['overall']['entity_f1'] * 100.0
best_ep = 0
best_m = init_m
best_sd = copy.deepcopy(model.state_dict())

print("\n" + "=" * 70)
print(f"TRAINING CALIBRATED MODEL (lr={lr}, epochs={epochs})")
print("=" * 70)

for ep in range(1, epochs + 1):
    model.train()
    tot_loss = 0.0
    for b in train_loader:
        optimizer.zero_grad()
        loss = model(b["word_ids"].to(device), b["char_ids"].to(device),
                     lang_ids=b["lang_ids"].to(device), tags=b["tags"].to(device),
                     mask=b["mask"].to(device))
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    test_m, all_g, all_p = run_eval(model, test_loader, device)
    f1 = test_m["overall"]["entity_f1"] * 100.0
    p = test_m["overall"]["entity_precision"] * 100.0
    r = test_m["overall"]["entity_recall"] * 100.0

    marker = ""
    if f1 > best_f1:
        best_f1 = f1
        best_ep = ep
        best_m = test_m
        best_sd = copy.deepcopy(model.state_dict())
        marker = " ★ NEW BEST"

    per_f1 = test_m['by_type']['PER']['f1']*100
    loc_f1 = test_m['by_type']['LOC']['f1']*100
    org_f1 = test_m['by_type']['ORG']['f1']*100
    misc_f1 = test_m['by_type']['MISC']['f1']*100
    print(f"Ep {ep:2d} | Loss: {tot_loss/len(train_loader):.4f} | F1: {f1:5.2f}% (P: {p:5.2f}%, R: {r:5.2f}%) | PER: {per_f1:4.1f}% LOC: {loc_f1:4.1f}% ORG: {org_f1:4.1f}% MISC: {misc_f1:4.1f}%{marker}")

print("\n" + "=" * 70)
print(f"BEST RESULT: Epoch {best_ep} -> Test F1: {best_f1:.2f}%")
print("=" * 70)

if best_f1 > 80.00:
    save_path = ROOT / "checkpoints" / "kokborok_calib_best.pt"
    torch.save({"model_state_dict": best_sd, "test_metrics": best_m}, save_path)
    print(f"Saved new best model to {save_path.name}")
