"""
Online Real-World Test Suite for 4-Language Robust NER.

Tests uncurated online and romanized/inflected sentences for:
- Konkani
- Assamese
- Sinhala
- Kokborok
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from deployment.inference import get_predictor
from preprocessing.multilingual_robustness import (
    detect_script,
    transliterate_to_native,
    MorphologicalDecoupler,
    GazetteerReconciler,
)
from training.train_charcnn import ID_TO_TAG
from evaluation.seqeval_metrics import extract_entities

TEST_CASES = {
    "konkani": [
        {
            "name": "Romanized Konkani with complex inflections & institutions",
            "sentence": "Rahul Naik 20 September 2026 disa Panajiant Goa Universityant ayojit kellya parishadent bhag gheta.",
            "expected_entities": ["PER", "LOC", "ORG"],
        },
        {
            "name": "Native Devanagari Konkani with locative inflections",
            "sentence": "प्रमोद सावंत गोंयचे मुख्य मंत्री आसात आनी तांणी पणजींत नव्या पुलाचे उद्घाटन केलें .",
            "expected_entities": ["PER", "LOC"],
        },
    ],
    "assamese": [
        {
            "name": "Native Assamese with institutional and locative case suffixes",
            "sentence": "হিমন্ত বিশ্ব শৰ্মা গুৱাহাটীত কটন বিশ্ববিদ্যালয়ত এখন সভাত অংশগ্ৰহণ কৰে ।",
            "expected_entities": ["PER", "LOC", "ORG"],
        },
        {
            "name": "Romanized Assamese online sentence",
            "sentence": "Himanta Biswa Sarma Guwahatit Cotton Universityt ekhon sobhat onkhogrohon kore.",
            "expected_entities": ["PER", "LOC", "ORG"],
        },
    ],
    "sinhala": [
        {
            "name": "Native Sinhala with institutional and locative suffixes",
            "sentence": "රනිල් වික්‍රමසිංහ කොළඹ පිහිටි කොළඹ විශ්වවිද්‍යාලයේ උත්සවයකට සහභාගී විය .",
            "expected_entities": ["PER", "LOC", "ORG"],
        },
        {
            "name": "Romanized Sinhala sentence",
            "sentence": "Ranil Wickremasinghe Colombo Colombu Vishwavidyalaye uthsawayakata sahbhagi wiya.",
            "expected_entities": ["PER", "LOC", "ORG"],
        },
    ],
    "kokborok": [
        {
            "name": "Kokborok sentence with personal name, locative clitic, and university",
            "sentence": "Biplab Kumar Deb Agartala-o Tripura University-ni kothoma-o phai kha.",
            "expected_entities": ["PER", "LOC", "ORG"],
        },
        {
            "name": "Kokborok sentence with autonomous council (TTAADC) and headquarters",
            "sentence": "Manik Saha Khumulwng-o TTAADC office-o thangkha.",
            "expected_entities": ["PER", "LOC", "ORG"],
        },
    ],
}


def run_robust_inference(predictor, sentence: str, language: str):
    decoupler = MorphologicalDecoupler(language, predictor.word_to_id)
    reconciler = GazetteerReconciler(language)

    # 1. Script normalization / transliteration if Roman input in non-Roman language
    script = detect_script(sentence)
    is_roman = (script == "latin")
    if is_roman and language != "kokborok":
        working_sentence = transliterate_to_native(sentence, language)
    else:
        working_sentence = sentence

    # 2. Tokenization
    raw_tokens = predictor.tokenizer.tokenize_words(working_sentence)

    # 3. Morphological Decoupling
    segmented_tokens = []
    for tok in raw_tokens:
        segmented_tokens.extend(decoupler.segment_word(tok))

    # 4. Neural Model Forward Pass
    w, c, l, mask = predictor.extractor.prepare_input_tensors(segmented_tokens)
    with torch.no_grad():
        paths = predictor.model(w, c, lang_ids=l, mask=mask)

    pred_tags = [ID_TO_TAG.get(tid, "O") for tid in paths[0][: len(segmented_tokens)]]
    raw_spans = extract_entities(pred_tags)

    pred_entities = []
    for etype, s_idx, e_idx in sorted(raw_spans, key=lambda x: (x[1], x[2])):
        pred_entities.append({
            "type": etype,
            "text": " ".join(segmented_tokens[s_idx:e_idx]),
            "start_token": s_idx,
            "end_token": e_idx,
        })

    # 5. Reconcile with Gazetteers & Patterns
    final_entities = reconciler.reconcile_entities(segmented_tokens, pred_entities)

    return {
        "is_roman": is_roman,
        "working_sentence": working_sentence,
        "tokens": segmented_tokens,
        "entities": final_entities,
    }


def main():
    print("=" * 80)
    print("AUTONOMOUS 4-LANGUAGE ROBUST NER EVALUATION ON UNCURATED ONLINE SENTENCES")
    print("=" * 80)

    for lang, test_items in TEST_CASES.items():
        print(f"\n[{lang.upper()}] Loading locked best model bundle...")
        predictor = get_predictor(lang)
        print(f"[{lang.upper()}] Predictor loaded successfully on {predictor.device}.\n")

        for idx, item in enumerate(test_items, 1):
            name = item["name"]
            sent = item["sentence"]
            print(f"--- Test Case {idx}: {name} ---")
            print(f"Input: {sent}")
            
            res = run_robust_inference(predictor, sent, lang)
            if res["is_roman"] and lang != "kokborok":
                print(f"Transliterated: {res['working_sentence']}")

            ents = res["entities"]
            print(f"Extracted Entities ({len(ents)}):")
            for ent in ents:
                print(f"  * [{ent['type']}] {ent['text']} (span {ent['start_token']}:{ent['end_token']})")
            print()


if __name__ == "__main__":
    main()
