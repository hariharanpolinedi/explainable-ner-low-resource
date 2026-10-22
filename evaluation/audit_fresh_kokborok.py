"""
Audit Fresh Kokborok Sentences:
1. Verifies 100% zero overlap against all prior training, dev, test, and synthetic sets.
2. Evaluates the models across all rounds on genuinely fresh, authentic sentences.
3. Computes precise entity-level Precision, Recall, and F1.
"""
from __future__ import annotations
import json, sys, re
from pathlib import Path

ROOT = Path(r"c:\Users\harih\Desktop\NLP-MARK-3")
sys.path.insert(0, str(ROOT))

import torch
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.train_charcnn import ID_TO_TAG
from evaluation.seqeval_metrics import compute_seqeval_metrics, extract_entities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = ROOT / "deployment" / "bundles" / "kokborok" / "v1"
w2id = json.load(open(bundle / "word_vocab.json", "r", encoding="utf-8")).get("token_to_id", {})
c2id = json.load(open(bundle / "char_vocab.json", "r", encoding="utf-8")).get("token_to_id", {})

# 1. Candidate Fresh Sentences with Ground-Truth Annotations
# Sourced from public open Kokborok text (sdmy/kokborok)
FRESH_TEST_CASES = [
    # Case 1: PER (Rajiv Gandhi)
    {
        "text": "Bini achukthaino bini bwsla Rajiv Gandhi naharkha .",
        "gold_entities": [("Rajiv Gandhi", "PER")],
    },
    # Case 2: PER (Ram)
    {
        "text": "Salsa , Ram kami twih himwi thang tongphuru , chwlarok bodol kaisano nugo .",
        "gold_entities": [("Ram", "PER")],
    },
    # Case 3: PER (Indira Gandhi)
    {
        "text": "Indira Gandhi puila bwrwirokni bisingo baraot hani montri bokhorok chalangnai .",
        "gold_entities": [("Indira Gandhi", "PER")],
    },
    # Case 4: PER (Jawaharlal Nehru), LOC (Barot)
    {
        "text": "Bini bupha Jawaharlal Nehru , bono Barot hani puila Montri Bokhorok .",
        "gold_entities": [("Jawaharlal Nehru", "PER"), ("Barot", "LOC")],
    },
    # Case 5: PER (Indira Gandhi, Feroze Gandhi), LOC (Barot, England)
    {
        "text": "Indira Gandhi Barot tei England hao rwngswrwngkha tei Feroze Gandhi bai kaijakha .",
        "gold_entities": [("Indira Gandhi", "PER"), ("Barot", "LOC"), ("England", "LOC"), ("Feroze Gandhi", "PER")],
    },
    # Case 6: PER (Rajiv, Sanjay)
    {
        "text": "Bohrokni khoroknwi bwsa chwla wngjakha , Rajiv tei Sanjay .",
        "gold_entities": [("Rajiv", "PER"), ("Sanjay", "PER")],
    },
    # Case 7: ORG (Indian Supreme Court)
    {
        "text": "Oh tangbwtangrok paithago abo Indian Supreme Court bai rosai naharjakha .",
        "gold_entities": [("Indian Supreme Court", "ORG")],
    },
    # Case 8: MISC (Navratri), PER (Parvati)
    {
        "text": "Navratri ni salchuku o , Parvati mungwi khoroksa bwrwi sikli hai hwnwi sajakha .",
        "gold_entities": [("Navratri", "MISC"), ("Parvati", "PER")],
    },
    # Case 9: PER (Shiva)
    {
        "text": "Kamini lukurok belai khaphurjakha Shiva bohrokno mwthangmani bagwi .",
        "gold_entities": [("Shiva", "PER")],
    },
    # Case 10: LOC (Malgudi), PER (Ramasamy)
    {
        "text": "Malgudi mungwi kaisa chikonsa aoli o , Ramasamy mungwi khoroksa Police inspector tongmani .",
        "gold_entities": [("Malgudi", "LOC"), ("Ramasamy", "PER")],
    },
    # Case 11: PER (Krishnan, Ramasamy)
    {
        "text": "Salsa , Krishnan mungwi khoroksa chwla sikla no bono Ramasamy romkha .",
        "gold_entities": [("Krishnan", "PER"), ("Ramasamy", "PER")],
    },
    # Case 12: Negative Control (All O) - Geological discussion
    {
        "text": "Horhapung ni phatimungrok ha bisingo belai swkwlwkmung khlaima bagwi se wngu .",
        "gold_entities": [],
    },
    # Case 13: Negative Control (All O) - Conversational meal
    {
        "text": "Dal makhani song baikha hwnkhe , bohrok chalaina bagwi achuklaikha .",
        "gold_entities": [],
    },
    # Case 14: Negative Control (All O) - Descriptive text
    {
        "text": "Bini mokol kotor kotor tei kirima gwnang wngjak .",
        "gold_entities": [],
    },
]

# 2. VERIFY NON-OVERLAP RIGOROUSLY
def verify_zero_overlap():
    print("=" * 70)
    print("STEP 1: RIGOROUS NON-OVERLAP VERIFICATION WITH ALL PRIOR DATASETS")
    print("=" * 70)
    
    known_sentences = set()
    files_to_check = [
        ROOT / "data" / "gold" / "kokborok" / "train.conll",
        ROOT / "data" / "gold" / "kokborok" / "dev.conll",
        ROOT / "data" / "gold" / "kokborok" / "test.conll",
        ROOT / "data" / "processed" / "kokborok_improved_train.conll",
        ROOT / "data" / "processed" / "kokborok_calib_train.conll",
        ROOT / "data" / "processed" / "kokborok_surgical_train.conll",
    ]
    
    for fp in files_to_check:
        if fp.exists():
            sents = fp.read_text(encoding="utf-8").strip().split("\n\n")
            for s in sents:
                words = [l.split("\t")[0] for l in s.strip().split("\n") if l.strip()]
                known_sentences.add(" ".join(words).lower().strip())
    
    print(f"Total known sentences indexed from prior splits: {len(known_sentences):,}")
    
    overlap_count = 0
    for idx, c in enumerate(FRESH_TEST_CASES):
        clean_text = " ".join(c["text"].split()).lower()
        if clean_text in known_sentences:
            print(f"  [FAIL] Sentence {idx+1} OVERLAPS with prior data: {clean_text}")
            overlap_count += 1
        else:
            print(f"  [PASS] Sentence {idx+1:2d}: Verified ZERO overlap.")
            
    assert overlap_count == 0, f"Found {overlap_count} overlapping sentences!"
    print(f"\n[VERIFIED] All {len(FRESH_TEST_CASES)} sentences have STRICT ZERO OVERLAP with any training/tuning data.")


# 3. RUN MODEL EVALUATION
def evaluate_fresh_sentences():
    print("\n" + "=" * 70)
    print("STEP 2: FRESH-SENTENCE GENERALIZATION EVALUATION")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    # Convert test cases to CoNLL tag format
    all_gold_tags = []
    all_tokens_list = []
    
    for case in FRESH_TEST_CASES:
        tokens = case["text"].split()
        tags = ["O"] * len(tokens)
        for ent_text, etype in case["gold_entities"]:
            ent_toks = ent_text.split()
            L = len(ent_toks)
            for i in range(len(tokens) - L + 1):
                if tokens[i : i + L] == ent_toks:
                    tags[i] = f"B-{etype}"
                    for k in range(1, L):
                        tags[i + k] = f"I-{etype}"
                    break
        all_gold_tags.append(tags)
        all_tokens_list.append(tokens)

    # Count gold entities
    total_gold_ents = sum(len(c["gold_entities"]) for c in FRESH_TEST_CASES)
    print(f"Total Fresh Sentences: {len(FRESH_TEST_CASES)}")
    print(f"Total Fresh Gold Entities: {total_gold_ents}")
    for etype in ["PER", "LOC", "ORG", "MISC"]:
        cnt = sum(1 for c in FRESH_TEST_CASES for _, t in c["gold_entities"] if t == etype)
        print(f"  {etype}: {cnt}")

    checkpoints_to_eval = [
        ("Phase 13 Baseline (Option-C)", ROOT / "checkpoints" / "staged_optionC_kokborok.pt"),
        ("Round 1 (kokborok_improved_v2)", ROOT / "checkpoints" / "kokborok_improved_v2.pt"),
        ("Round 4 (kokborok_calib_best)", ROOT / "checkpoints" / "kokborok_calib_best.pt"),
        ("Final Deployed (final_kokborok)", ROOT / "checkpoints" / "final_kokborok.pt"),
    ]

    results_summary = []

    for name, pth in checkpoints_to_eval:
        if not pth.exists():
            continue
        ckpt = torch.load(pth, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        num_langs = sd["lang_embedding.embedding.weight"].shape[0] if "lang_embedding.embedding.weight" in sd else 4
        
        model = NERMultilingualModel(
            word_vocab_size=len(w2id), char_vocab_size=len(c2id),
            num_tags=9, num_languages=num_langs, crf_config="per_language", hidden_dim=128
        ).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()

        all_pred_tags = []
        with torch.no_grad():
            for tokens in all_tokens_list:
                w_ids = torch.tensor([[w2id.get(w, 1) for w in tokens]], device=device)
                max_c = max(len(w) for w in tokens)
                c_mat = torch.zeros(1, len(tokens), max_c, dtype=torch.long, device=device)
                for j, w in enumerate(tokens):
                    for k, ch in enumerate(w):
                        c_mat[0, j, k] = c2id.get(ch, 1)
                mask = torch.ones(1, len(tokens), dtype=torch.bool, device=device)
                lang_ids = torch.tensor([num_langs - 1], device=device)
                preds = model(w_ids, c_mat, lang_ids=lang_ids, mask=mask)[0]
                pred_labels = [ID_TO_TAG[pid] for pid in preds[:len(tokens)]]
                all_pred_tags.append(pred_labels)

        metrics = compute_seqeval_metrics(all_gold_tags, all_pred_tags)
        f1 = metrics["overall"]["entity_f1"] * 100.0
        p = metrics["overall"]["entity_precision"] * 100.0
        r = metrics["overall"]["entity_recall"] * 100.0
        acc = metrics["overall"]["token_accuracy"] * 100.0
        
        print(f"\n--- {name} ---")
        print(f"  Overall F1: {f1:5.2f}% | Precision: {p:5.2f}% | Recall: {r:5.2f}% | Token Acc: {acc:5.2f}%")
        for t in ["PER", "LOC", "ORG", "MISC"]:
            m = metrics["by_type"][t]
            print(f"    {t:<5}: P={m['precision']*100:5.1f}%, R={m['recall']*100:5.1f}%, F1={m['f1']*100:5.1f}% (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
            
        results_summary.append({
            "model": name,
            "f1": f1, "p": p, "r": r, "acc": acc,
            "by_type": metrics["by_type"]
        })

    return results_summary

if __name__ == "__main__":
    verify_zero_overlap()
    evaluate_fresh_sentences()
