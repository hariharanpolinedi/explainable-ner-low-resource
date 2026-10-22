"""
Zero-Gazetteer Fictional Entities Test.
Tests whether the raw neural weights can identify completely invented, novel entities
that exist in NO dictionary or gazetteer on Earth.
"""

import sys
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
from evaluation.seqeval_metrics import extract_entities
from training.train_charcnn import ID_TO_TAG


def test_fictional():
    print("=" * 80)
    print("TESTING PURE NEURAL GENERALIZATION ON 100% FICTIONAL / UNSEEN ENTITIES")
    print("Zero Gazetteers, Zero Dictionary Lookups. Pure Neural Weights (BiLSTM + Char-CNN + CRF).")
    print("=" * 80)

    predictor = get_predictor("kokborok")

    test_cases = [
        {
            "desc": "Fictional Startup & Unseen Person in Rural Town",
            "sentence": "Dr. Tanmoy Sengupta-ni 14 March 2028-ot Teliamura-ao NexaTech Labs aru DeepMind-ni logot discussion khorok.",
            "unseen_entities": ["Dr. Tanmoy Sengupta", "14 March 2028", "Teliamura", "NexaTech Labs", "DeepMind"],
        },
        {
            "desc": "Fictional Autonomous Institute & Novel Person",
            "sentence": "Professor Bikash Debnath-ni 19 August 2029-ot Sabroom-ao CyberDyn Robotics aru QuantumDeep AI-ni research seminar-ot jog dai.",
            "unseen_entities": ["Professor Bikash Debnath", "19 August 2029", "Sabroom", "CyberDyn Robotics", "QuantumDeep AI"],
        },
        {
            "desc": "Invented Research Project & Novel Location",
            "sentence": "24 November 2028-ot Belonia-ao Khowai Technologies aru BioSynth Labs-ni scientists-mani speech synthesis aru neuromorphic computing-ni upor training suru khorok.",
            "unseen_entities": ["24 November 2028", "Belonia", "Khowai Technologies", "BioSynth Labs", "speech synthesis", "neuromorphic computing"],
        },
    ]

    for tc in test_cases:
        print(f"\n[SCENARIO] {tc['desc']}")
        sent = tc["sentence"]
        print(f"Sentence: \"{sent}\"")
        print(f"Unseen Ground Truths: {tc['unseen_entities']}")

        # Pure Neural Forward Pass (no reconciler)
        tokens = predictor.tokenizer.tokenize_words(sent)
        w, c, l, mask = predictor.extractor.prepare_input_tensors(tokens)
        with torch.no_grad():
            paths = predictor.model(w, c, lang_ids=l, mask=mask)

        pred_tags = [ID_TO_TAG.get(tid, "O") for tid in paths[0][: len(tokens)]]
        raw_spans = extract_entities(pred_tags)

        print("  -> Raw Neural Predictions:")
        for etype, s_idx, e_idx in sorted(raw_spans, key=lambda x: (x[1], x[2])):
            span_text = " ".join(tokens[s_idx:e_idx])
            print(f"     * [{etype}] \"{span_text}\" (tokens {s_idx}..{e_idx})")


if __name__ == "__main__":
    test_fictional()
