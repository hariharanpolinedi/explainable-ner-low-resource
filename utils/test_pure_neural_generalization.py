"""
Strict Pure Neural Generalization Test (Zero Gazetteers, Zero Rules, Zero Dictionaries).
Evaluates raw neural network forward pass (Char-CNN + BiLSTM + CRF) on the 10 sentences
to measure what the neural weights learned from backpropagation alone.
"""

import sys
import json
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from deployment.inference import get_predictor
from utils.test_10_new_kokborok_sentences import DATASET_10, clean_text
from evaluation.seqeval_metrics import extract_entities
from training.train_charcnn import ID_TO_TAG


def evaluate_pure_neural():
    print("=" * 80)
    print("STRICT PURE NEURAL GENERALIZATION TEST (ZERO GAZETTEERS, ZERO RULES)")
    print("Testing RAW neural network forward pass (BiLSTM + Char-CNN + CRF) on GPU")
    print("=" * 80)

    # Force re-loading fresh weights
    from deployment import inference
    inference._PREDICTOR_CACHE.clear()

    predictor = get_predictor("kokborok")
    total_expected = 0
    total_matched = 0

    for item in DATASET_10:
        sid = item["id"]
        title = item["title"]
        sent = item["sentence"]
        expected = item["expected_entities"]
        total_expected += len(expected)

        # 1. Tokenize words
        tokens = predictor.tokenizer.tokenize_words(sent)

        # 2. Prepare raw tensors for neural network forward pass
        w, c, l, mask = predictor.extractor.prepare_input_tensors(tokens)
        with torch.no_grad():
            # Raw Viterbi decoding directly from neural CRF
            paths = predictor.model(w, c, lang_ids=l, mask=mask)

        pred_tags = [ID_TO_TAG.get(tid, "O") for tid in paths[0][: len(tokens)]]

        # 3. Extract entities strictly from raw neural CRF tag sequence
        raw_spans = extract_entities(pred_tags)
        raw_neural_entities = []
        for etype, s_idx, e_idx in sorted(raw_spans, key=lambda x: (x[1], x[2])):
            raw_neural_entities.append({
                "type": etype,
                "text": " ".join(tokens[s_idx:e_idx]),
                "start_token": s_idx,
                "end_token": e_idx,
            })

        matched = []
        missed = []
        for exp_text, exp_type in expected:
            clean_exp = clean_text(exp_text)
            found = False
            for p in raw_neural_entities:
                clean_pred = clean_text(p["text"])
                if (clean_exp == clean_pred or clean_exp in clean_pred or clean_pred in clean_exp) and exp_type == p["type"]:
                    matched.append((exp_text, exp_type, p["text"]))
                    found = True
                    break
            if not found:
                missed.append((exp_text, exp_type))

        total_matched += len(matched)
        recall = (len(matched) / len(expected) * 100) if expected else 0.0
        prec = (len(matched) / len(raw_neural_entities) * 100) if raw_neural_entities else 0.0

        print(f"\n[{sid}] {title}")
        print(f"  * Expected Entities: {len(expected)}")
        print(f"  * RAW Neural Extracted: {len(raw_neural_entities)}")
        print(f"  * Correctly Matched by Neural Weights Alone: {len(matched)} ({recall:.1f}% Recall | {prec:.1f}% Precision)")
        if missed:
            print(f"  * Missed by Neural Alone ({len(missed)}): {[(m[0], m[1]) for m in missed]}")
        else:
            print("  * Missed: 0 (FLAWLESS PURE NEURAL EXTRACTION)")

    overall_recall = (total_matched / total_expected * 100) if total_expected else 0.0
    print("\n" + "=" * 80)
    print(f"PURE NEURAL GENERALIZATION SCORE: {total_matched} / {total_expected} ({overall_recall:.1f}% Recall)")
    print("Zero gazetteers used. All predictions generated purely by BiLSTM-CRF neural weights.")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_pure_neural()
