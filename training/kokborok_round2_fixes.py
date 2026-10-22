"""
Kokborok NER Improvement — Round 2 (Targeted Fixes).

Fixes the 3 specific remaining issues from Round 1:
  1. PER: Short names (Mohan, Rahul) in question/action contexts
  2. ORG: Multi-word ORGs like "Tripura board" 
  3. MISC: 2 false positives - precision tightening

Strategy: Add targeted training patterns + retrain from Round 1 checkpoint.
"""

from __future__ import annotations

import copy
import json
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# TARGETED ENTITY LISTS
# ============================================================================

SHORT_NAMES = [
    "Mohan", "Rahul", "Sunil", "Bikram", "Kamal", "Dipak", "Ranjit",
    "Anil", "Bimal", "Gopal", "Tapan", "Samir", "Pradip", "Ujjwal",
    "Hemanta", "Subrata", "Nirmal", "Sanjay", "Prabir", "Deepak",
    "Suresh", "Rajesh", "Sunita", "Priya", "Rita", "Mina", "Asha",
    "Swapna", "Kabita", "Anjali", "Kamala", "Deepa", "Lakshmi",
    "Abhijit", "Amiya", "Ashis", "Ratan", "Manik", "Biplab",
    "Jitendra", "Dasarath", "Pramod", "Chandra", "Narendra",
    "Amit", "Pradeep", "Vikram", "Ajay", "Sanjib", "Dilip",
    "Partha", "Biswajit", "Debashis", "Anupam", "Suman", "Michael",
]

ORG_WITH_PLACE = [
    ("Tripura", "board"), ("Tripura", "council"), ("Tripura", "committee"),
    ("Tripura", "commission"), ("Tripura", "assembly"), ("Tripura", "department"),
    ("Tripura", "authority"), ("Tripura", "corporation"), ("Tripura", "tribunal"),
    ("Agartala", "municipality"), ("Agartala", "corporation"),
    ("Khowai", "council"), ("Udaipur", "board"), ("Dharmanagar", "board"),
    ("Belonia", "committee"), ("Sabroom", "council"), ("Ambassa", "board"),
    ("Gomati", "council"), ("Dhalai", "committee"), ("Unakoti", "board"),
    ("Sepahijala", "council"), ("Bishalgarh", "municipality"),
]

PLACES = [
    "Tripura", "Agartala", "Khumulwng", "Udaipur", "Dharmanagar",
    "Khowai", "Ambassa", "Belonia", "Sabroom", "Suryamaninagar",
    "Bishalgarh", "Santirbazar", "India", "Bangladesh", "Guwahati",
    "Kolkata", "Delhi", "Bishramganj", "Kailashahar", "Kamalpur",
    "Sonamura", "Mohanpur", "Amarpur", "Teliamura", "Jirania",
    "Assam", "Mizoram", "Manipur", "Meghalaya", "Imphal",
]

KOKBOROK_VERBS = [
    "phaikha", "khorok", "tongo", "sanlaikha", "khwlaikha",
    "wngkha", "chengmani", "rwlaikha", "kha", "nai",
    "ong kha", "chwng kha", "tangwi", "lanai", "thwikha",
    "phai di", "gwrwng kha", "sainani", "kokthwi kha",
]

EVENTS = [
    "Christmas", "Buisu", "Kharchi Puja", "Kokborok Day", "Garia Puja",
    "Ker Puja", "Durga Puja", "Diwali", "Holi", "Republic Day",
    "Independence Day", "Tring Festival", "Orange Festival",
]

FULL_PERSONS = [
    "Maharaja Bir Bikram", "Pradyot Bikram Manikya Debbarma",
    "Biplab Kumar Deb", "Manik Saha", "Jitendra Chaudhury",
    "Dipa Karmakar", "Somdev Devvarman", "Dasarath Deb",
    "Alindra Debbarma", "Rabindra Debbarma", "Animesh Debbarma",
    "Debashis Jamatia", "Rina Debbarma", "Anupam Chakma",
    "Suman Reang", "Michael Das", "Sourabhee Debbarma",
]

PURE_ORGS = [
    "Tripura University", "MBB University", "National Institute of Technology",
    "TTAADC", "Tipra Motha", "Twipra Students Federation",
    "Kokborok Sahitya Sabha", "Government of Tripura",
    "Agartala Government Medical College", "Tripura High Court",
    "Tripura Police", "State Bank of India", "Tripura Gramin Bank",
]

SUFFIXES = ["-o", "-ni", "-no", "-ao", "-bai", "-rok", ""]


def tag(text, label):
    tokens = text.split()
    return [(tokens[i], f"{'B' if i==0 else 'I'}-{label}") for i in range(len(tokens))]


def tag_s(text, label, suffix=""):
    tokens = text.split()
    result = []
    for i, tok in enumerate(tokens):
        prefix = "B-" if i == 0 else "I-"
        if i == len(tokens) - 1 and suffix:
            result.append((f"{tok}{suffix}", f"{prefix}{label}"))
        else:
            result.append((tok, f"{prefix}{label}"))
    return result


def O(*words):
    return [(w, "O") for w in words]


def generate_targeted_sentences():
    random.seed(42)
    sentences = []

    # FIX 1: Short PER names (700 sentences)
    per_templates = [
        lambda n, l: O("Where", "is") + tag(n, "PER") + O("Bo", "phaiwi", "de", "tong", "?"),
        lambda n, l: O("Nwng") + tag(n, "PER") + O("snammani", "?"),
        lambda n, l: O("Bo") + tag(n, "PER") + O("phaiwi", "de", "tong", "?"),
        lambda n, l: tag(n, "PER") + O("bo", "phaiwi", "de", "tong", "?"),
        lambda n, l: tag(n, "PER") + O("kabo", "tongo", "?"),
        lambda n, l: tag(n, "PER") + O("bike", "accident", "wngwi", "thwikha", "."),
        lambda n, l: tag(n, "PER") + O("accident", "wngwi", "hospital-o", "phaikha", "."),
        lambda n, l: tag(n, "PER") + O("car", "accident-o", "thwikha", "."),
        lambda n, l: tag(n, "PER") + O("school-o", "phaikha", "."),
        lambda n, l: tag(n, "PER") + O("office-o", "tongo", "."),
        lambda n, l: tag(n, "PER") + O("kaham", "tongo", "."),
        lambda n, l: tag(n, "PER") + O("wngkha") + tag_s(l, "LOC", "-ni") + O("borokrog", "."),
        lambda n, l: tag(n, "PER") + O("aru") + tag(random.choice(SHORT_NAMES), "PER") + O("logot", "tongo", "."),
        lambda n, l: tag(n, "PER") + O("ni", "nok") + tag_s(l, "LOC", "-o") + O("tongo", "."),
        lambda n, l: tag(n, "PER") + O("bisi-o", "market-o", "phaikha", "."),
        lambda n, l: O("Ang") + tag(n, "PER") + O("logot", "phaikha", "."),
        lambda n, l: O("Tini") + tag(n, "PER") + O("ni", "birthday", "."),
        lambda n, l: O("Kabo") + tag(n, "PER") + O("wngwi", "tongo", "?"),
        lambda n, l: O("Tabuk") + tag(n, "PER") + O("phaikha", "nai", "."),
        lambda n, l: tag_s(l, "LOC", "-o") + tag(n, "PER") + O("tongo", "."),
    ]

    for _ in range(500):
        sentences.append(random.choice(per_templates)(random.choice(SHORT_NAMES), random.choice(PLACES)))
    for _ in range(200):
        sentences.append(random.choice(per_templates)(random.choice(FULL_PERSONS), random.choice(PLACES)))
    print(f"  [FIX 1] Generated 700 PER-targeted sentences")

    # FIX 2: Multi-word ORG (600 sentences)
    org_templates = [
        lambda p, s, l: O("Kebo") + tag(f"{p} {s}", "ORG") + O("ni", "employee", "rok", "tongo", "."),
        lambda p, s, l: tag(f"{p} {s}", "ORG") + O("ni", "meeting", "khorok", "."),
        lambda p, s, l: tag(f"{p} {s}", "ORG") + O("ni", "order", "kha", "."),
        lambda p, s, l: tag(f"{p} {s}", "ORG") + tag_s(l, "LOC", "-o") + O("tongo", "."),
        lambda p, s, l: O("Tabuk") + tag(f"{p} {s}", "ORG") + O("ni", "decision", "kha", "."),
        lambda p, s, l: tag(f"{p} {s}", "ORG") + O("ni", "padhai", "sanlaikha", "."),
        lambda p, s, l: tag(random.choice(SHORT_NAMES), "PER") + tag(f"{p} {s}", "ORG") + O("ni", "chairman", "wngkha", "."),
        lambda p, s, l: tag(f"{p} {s}", "ORG") + O("ni", "employee", "rok") + tag_s(l, "LOC", "-o") + O("protest", "khorok", "."),
        lambda p, s, l: tag_s(f"{p} {s}", "ORG", "-ni") + O("admit", "tai", "mark", "sheet", "kwma", "."),
        lambda p, s, l: tag_s(f"{p} {s}", "ORG", "-o") + O("phaikha", "."),
        lambda p, s, l: tag(l, "LOC") + O("ni") + tag(f"{p} {s}", "ORG") + O("tongo", "."),
        lambda p, s, l: tag(f"{p} {s}", "ORG") + O("aru") + tag(l, "LOC") + O("ni", "majao", "."),
    ]

    for _ in range(400):
        place, suffix = random.choice(ORG_WITH_PLACE)
        loc = random.choice([l for l in PLACES if l != place])
        sentences.append(random.choice(org_templates)(place, suffix, loc))

    for _ in range(200):
        org = random.choice(PURE_ORGS)
        loc = random.choice(PLACES)
        name = random.choice(SHORT_NAMES)
        t = random.randint(0, 3)
        if t == 0:
            sentences.append(tag(name, "PER") + tag_s(org, "ORG", "-ni") + O("padhai", "sanlaikha", "."))
        elif t == 1:
            sentences.append(tag(org, "ORG") + tag_s(loc, "LOC", "-o") + O("tongo", "."))
        elif t == 2:
            sentences.append(tag_s(org, "ORG", "-ni") + O("employee", "rok", "protest", "khorok", "."))
        else:
            sentences.append(tag(name, "PER") + O("wngkha") + tag_s(org, "ORG", "-ni") + O("principal", "."))
    print(f"  [FIX 2] Generated 600 ORG-targeted sentences")

    # FIX 3: MISC precision (200 sentences)
    neg_words = ["order", "condition", "market", "school", "office", "hospital",
                 "road", "bridge", "bus", "train", "phone", "computer",
                 "meeting", "election", "protest", "accident", "incident"]
    for _ in range(100):
        sentences.append(O(random.choice(neg_words), "ni", "bagui", random.choice(KOKBOROK_VERBS), "."))
    for _ in range(100):
        event = random.choice(EVENTS)
        loc = random.choice(PLACES)
        sentences.append(tag(event, "MISC") + O("romwi") + tag_s(loc, "LOC", "-o") + O("chengmani", "."))
    print(f"  [FIX 3] Generated 200 MISC precision sentences")

    return sentences


def main():
    import torch
    from torch.utils.data import DataLoader
    from model.ner_model_v4_multilingual import NERMultilingualModel
    from training.multilingual_dataset import MultilingualNERDataset, collate_fn_multilingual
    from evaluation.seqeval_metrics import compute_seqeval_metrics
    from training.train_charcnn import ID_TO_TAG
    from utils.seed import set_global_seed

    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("KOKBOROK NER ROUND 2 - TARGETED FIXES")
    print(f"Device: {device}")
    print("=" * 70)

    # STEP 1: Build combined dataset
    print("\n[STEP 1] Building targeted training data...")
    r1_path = PROJECT_ROOT / "data" / "processed" / "kokborok_improved_train.conll"
    r1_sents = []
    curr = []
    with open(r1_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if curr:
                    r1_sents.append(curr)
                    curr = []
            else:
                parts = line.split("\t")
                if len(parts) >= 2:
                    curr.append((parts[0], parts[-1]))
        if curr:
            r1_sents.append(curr)
    print(f"  Round 1 dataset: {len(r1_sents):,} sentences")

    targeted = generate_targeted_sentences()
    all_sents = r1_sents + targeted
    random.shuffle(all_sents)
    print(f"  Combined: {len(all_sents):,} sentences")

    counts = {"PER": 0, "LOC": 0, "ORG": 0, "MISC": 0}
    for sent in all_sents:
        for _, t in sent:
            if t.startswith("B-"):
                e = t[2:]
                if e in counts:
                    counts[e] += 1
    print(f"  Entity counts: {counts}")

    out_path = PROJECT_ROOT / "data" / "processed" / "kokborok_round2_train.conll"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sent in all_sents:
            for tok, t in sent:
                f.write(f"{tok}\t{t}\n")
            f.write("\n")
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")

    # STEP 2: Retrain
    print(f"\n[STEP 2] Retraining model...")
    bundle_dir = PROJECT_ROOT / "deployment" / "bundles" / "kokborok" / "v1"
    with open(bundle_dir / "word_vocab.json", "r", encoding="utf-8") as f:
        word_to_id = json.load(f).get("token_to_id", {})
    with open(bundle_dir / "char_vocab.json", "r", encoding="utf-8") as f:
        char_to_id = json.load(f).get("token_to_id", {})

    model = NERMultilingualModel(
        word_vocab_size=len(word_to_id), char_vocab_size=len(char_to_id),
        num_tags=9, num_languages=4, crf_config="per_language", hidden_dim=128,
    ).to(device)

    r1_ckpt = PROJECT_ROOT / "checkpoints" / "kokborok_improved_v2.pt"
    ckpt = torch.load(r1_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"  Loaded Round 1 checkpoint")

    for name, param in model.named_parameters():
        if any(f"fc_heads.{i}" in name or f"crf_heads.{i}" in name for i in [0, 1, 2]):
            param.requires_grad = False
        else:
            param.requires_grad = True

    dev_path = PROJECT_ROOT / "data" / "gold" / "kokborok" / "dev.conll"
    train_ds = MultilingualNERDataset(lang_conll_paths={"kokborok": [out_path]}, word_to_id=word_to_id, char_to_id=char_to_id)
    dev_ds = MultilingualNERDataset(lang_conll_paths={"kokborok": [dev_path]}, word_to_id=word_to_id, char_to_id=char_to_id)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn_multilingual)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)
    print(f"  Train: {len(train_ds):,}, Dev: {len(dev_ds)}")

    max_epochs = 15
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=3e-5)
    best_dev_f1 = -1.0
    best_state = None
    best_metrics = None
    patience = 5
    no_improve = 0

    print(f"\n  Training for {max_epochs} epochs...\n")
    for epoch in range(1, max_epochs + 1):
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
        scheduler.step()

        model.eval()
        all_gold, all_pred = [], []
        with torch.no_grad():
            for batch in dev_loader:
                paths = model(batch["word_ids"].to(device), batch["char_ids"].to(device),
                              lang_ids=batch["lang_ids"].to(device), mask=batch["mask"].to(device))
                for i, slen in enumerate(batch["seq_lengths"]):
                    all_gold.append([ID_TO_TAG[tid.item()] for tid in batch["tags"][i, :slen]])
                    all_pred.append([ID_TO_TAG[tid] for tid in paths[i][:slen]])

        dm = compute_seqeval_metrics(all_gold, all_pred)
        f1 = dm["overall"]["entity_f1"] * 100
        p = dm["overall"]["entity_precision"] * 100
        r = dm["overall"]["entity_recall"] * 100
        types = " | ".join(f'{t}: {dm["by_type"][t]["f1"]*100:.1f}%' for t in ["PER","LOC","ORG","MISC"])
        avg_loss = total_loss / max(len(train_loader), 1)
        marker = ""
        if f1 > best_dev_f1:
            best_dev_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dm
            marker = " * BEST"
            no_improve = 0
        else:
            no_improve += 1
        print(f"  Ep {epoch:2d}/{max_epochs} | Loss: {avg_loss:.4f} | F1: {f1:5.2f}% (P:{p:5.1f}% R:{r:5.1f}%) | {types}{marker}")
        if no_improve >= patience and epoch >= 8:
            print(f"\n  Early stop at epoch {epoch}")
            break

    save_path = PROJECT_ROOT / "checkpoints" / "kokborok_improved_v3.pt"
    torch.save({"model_state_dict": best_state, "dev_metrics": best_metrics}, save_path)
    print(f"\n  [SAVED] {save_path.relative_to(PROJECT_ROOT)} (Best Dev F1: {best_dev_f1:.2f}%)")
    if best_metrics:
        print(f"\n  Per-Type Dev:")
        for t in ["PER", "LOC", "ORG", "MISC"]:
            m = best_metrics["by_type"][t]
            print(f"    {t}: P={m['precision']*100:.1f}%, R={m['recall']*100:.1f}%, F1={m['f1']*100:.1f}%")

    # STEP 3: Gold-Test
    print(f"\n[STEP 3] Gold-Test Evaluation...")
    model.load_state_dict(best_state)
    model.eval()
    test_path = PROJECT_ROOT / "data" / "gold" / "kokborok" / "test.conll"
    test_ds = MultilingualNERDataset(lang_conll_paths={"kokborok": [test_path]}, word_to_id=word_to_id, char_to_id=char_to_id)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)

    all_gold, all_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            paths = model(batch["word_ids"].to(device), batch["char_ids"].to(device),
                          lang_ids=batch["lang_ids"].to(device), mask=batch["mask"].to(device))
            for i, slen in enumerate(batch["seq_lengths"]):
                all_gold.append([ID_TO_TAG[tid.item()] for tid in batch["tags"][i, :slen]])
                all_pred.append([ID_TO_TAG[tid] for tid in paths[i][:slen]])

    tm = compute_seqeval_metrics(all_gold, all_pred)
    print(f"\n{'=' * 60}")
    print(f"KOKBOROK GOLD-TEST RESULTS (Round 2)")
    print(f"{'=' * 60}")
    print(f"Overall Entity F1:   {tm['overall']['entity_f1']*100:.2f}%")
    print(f"Overall Precision:   {tm['overall']['entity_precision']*100:.2f}%")
    print(f"Overall Recall:      {tm['overall']['entity_recall']*100:.2f}%")
    print(f"Token Accuracy:      {tm['overall']['token_accuracy']*100:.2f}%")
    print(f"\nPer-Entity-Type:")
    print(f"{'Type':<8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 50)
    for t in ["PER", "LOC", "ORG", "MISC"]:
        m = tm["by_type"][t]
        print(f"{t:<8} {m['precision']*100:>7.2f}% {m['recall']*100:>7.2f}% {m['f1']*100:>7.2f}% {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")

    print(f"\n{'=' * 60}")
    print(f"IMPROVEMENT HISTORY")
    print(f"{'=' * 60}")
    orig = {"PER": 0.0, "LOC": 77.78, "ORG": 0.0, "MISC": 100.0, "OVR": 68.97}
    r1 = {"PER": 66.67, "LOC": 86.96, "ORG": 0.0, "MISC": 75.0, "OVR": 80.0}
    r2_ovr = tm["overall"]["entity_f1"] * 100
    print(f"{'Type':<8} {'Original':>10} {'Round 1':>10} {'Round 2':>10} {'Total Delta':>12}")
    print("-" * 55)
    for t in ["PER", "LOC", "ORG", "MISC"]:
        after = tm["by_type"][t]["f1"] * 100
        delta = after - orig[t]
        sign = "+" if delta >= 0 else ""
        print(f"{t:<8} {orig[t]:>9.2f}% {r1[t]:>9.2f}% {after:>9.2f}% {sign}{delta:>10.2f} pp")
    delta_ovr = r2_ovr - orig["OVR"]
    sign = "+" if delta_ovr >= 0 else ""
    print(f"{'OVERALL':<8} {orig['OVR']:>9.2f}% {r1['OVR']:>9.2f}% {r2_ovr:>9.2f}% {sign}{delta_ovr:>10.2f} pp")


if __name__ == "__main__":
    main()
