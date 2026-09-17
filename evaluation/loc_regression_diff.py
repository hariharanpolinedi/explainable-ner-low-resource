"""
LOC Regression Diff on Locked Gold-Dev.

Diffs baseline (checkpoints/augmented_sinhala.pt) vs conjunct_fixed (checkpoints/conjunct_fixed_sinhala.pt)
sentence-by-sentence on data/gold/sinhala/dev.conll.

Identifies:
1. Every LOC span where predictions differ between baseline and conjunct_fixed.
2. All False Positives (FP) and False Negatives (FN) for LOC in both models.
3. Sentence index, token sequence, gold tags, baseline predicted tags, and conjunct_fixed predicted tags.
4. Cross-checks against data/augmented/sinhala/conjunct_robustness.conll to see if:
   - Tokens in the error spans contain conjuncts.
   - Tokens appear in conjunct_robustness.conll with boundary or label discrepancies.
   - The error is a boundary error (partial match) or a spurious false-positive.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from evaluation.seqeval_metrics import extract_entities
from model.ner_model_v4_multilingual import NERMultilingualModel
from preprocessing.sinhala_conjunct_repair import decompose_sinhala_conjuncts
from training.multilingual_dataset import (
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from training.train_charcnn import ID_TO_TAG


def load_model(ckpt_path: Path, device: torch.device, w_to_id: dict, c_to_id: dict) -> NERMultilingualModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model = NERMultilingualModel(
        word_vocab_size=len(w_to_id),
        char_vocab_size=len(c_to_id),
        num_tags=9,
        num_languages=3,
        hidden_dim=128,
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def get_predictions_and_tokens(
    model: NERMultilingualModel,
    dev_path: Path,
    w_to_id: dict,
    c_to_id: dict,
    device: torch.device,
) -> list[dict[str, Any]]:
    ds = MultilingualNERDataset(
        lang_conll_paths={"sinhala": [dev_path]},
        word_to_id=w_to_id,
        char_to_id=c_to_id,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)

    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)
            seq_lengths = batch["seq_lengths"]
            tokens = batch["tokens"]

            paths = model(word_ids, char_ids, lang_ids=lang_ids, mask=mask)

            for i in range(len(seq_lengths)):
                slen = seq_lengths[i]
                sent_tokens = tokens[i][:slen]
                gold_seq = [ID_TO_TAG[t.item()] for t in tags[i, :slen]]
                pred_seq = [ID_TO_TAG[t] for t in paths[i][:slen]]
                records.append({
                    "tokens": sent_tokens,
                    "gold": gold_seq,
                    "pred": pred_seq,
                })

    return records


def load_augmented_corpus_tokens(aug_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Index tokens and their tags from conjunct_robustness.conll."""
    token_to_occurrences: dict[str, list[dict[str, Any]]] = {}
    current_sent: list[tuple[str, str]] = []

    with open(aug_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            l = line.strip()
            if not l:
                if current_sent:
                    sent_tokens = [t for t, _ in current_sent]
                    for idx, (t, tag) in enumerate(current_sent):
                        token_to_occurrences.setdefault(t, []).append({
                            "tag": tag,
                            "context": " ".join(sent_tokens[max(0, idx - 2): min(len(sent_tokens), idx + 3)]),
                            "line": line_num,
                        })
                    current_sent = []
            else:
                parts = l.split("\t")
                if len(parts) >= 2:
                    current_sent.append((parts[0], parts[1]))
                elif len(parts) == 1:
                    current_sent.append((parts[0], "O"))

    if current_sent:
        sent_tokens = [t for t, _ in current_sent]
        for idx, (t, tag) in enumerate(current_sent):
            token_to_occurrences.setdefault(t, []).append({
                "tag": tag,
                "context": " ".join(sent_tokens[max(0, idx - 2): min(len(sent_tokens), idx + 3)]),
                "line": line_num,
            })

    return token_to_occurrences


def analyze_loc_diff() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    gold_dev = PROJECT_ROOT / "data" / "gold" / "sinhala" / "dev.conll"
    aug_path = PROJECT_ROOT / "data" / "augmented" / "sinhala" / "conjunct_robustness.conll"
    base_ckpt = PROJECT_ROOT / "checkpoints" / "augmented_sinhala.pt"
    fixed_ckpt = PROJECT_ROOT / "checkpoints" / "conjunct_fixed_sinhala.pt"

    w_to_id, c_to_id = load_or_create_multilingual_vocabs(proc_dir)

    print("Loading models and evaluating sentences...")
    base_model = load_model(base_ckpt, device, w_to_id, c_to_id)
    base_records = get_predictions_and_tokens(base_model, gold_dev, w_to_id, c_to_id, device)

    fixed_model = load_model(fixed_ckpt, device, w_to_id, c_to_id)
    fixed_records = get_predictions_and_tokens(fixed_model, gold_dev, w_to_id, c_to_id, device)

    aug_tokens = load_augmented_corpus_tokens(aug_path)

    print("\n" + "=" * 90)
    print("SENTENCE-BY-SENTENCE LOC REGRESSION DIFF (Baseline vs Conjunct-Fixed)")
    print("=" * 90)

    flipped_sentences = []
    all_loc_errors = []

    for s_idx, (b_rec, f_rec) in enumerate(zip(base_records, fixed_records)):
        tokens = b_rec["tokens"]
        golds = b_rec["gold"]
        b_preds = b_rec["pred"]
        f_preds = f_rec["pred"]

        gold_entities = extract_entities(golds)
        b_entities = extract_entities(b_preds)
        f_entities = extract_entities(f_preds)

        gold_locs = {e for e in gold_entities if e[0] == "LOC"}
        b_locs = {e for e in b_entities if e[0] == "LOC"}
        f_locs = {e for e in f_entities if e[0] == "LOC"}

        # Check if LOC predictions differ
        if b_locs != f_locs:
            flipped_sentences.append({
                "sent_idx": s_idx,
                "tokens": tokens,
                "gold": golds,
                "gold_locs": gold_locs,
                "b_preds": b_preds,
                "b_locs": b_locs,
                "f_preds": f_preds,
                "f_locs": f_locs,
            })

    print(f"Total dev sentences: {len(base_records)}")
    print(f"Sentences where LOC prediction changed: {len(flipped_sentences)}")
    print("-" * 90)

    for case in flipped_sentences:
        s_idx = case["sent_idx"]
        tokens = case["tokens"]
        gold_locs = case["gold_locs"]
        b_locs = case["b_locs"]
        f_locs = case["f_locs"]

        print(f"\n[Sentence #{s_idx + 1}]")
        print(f"Full Text: {' '.join(tokens)}")
        print(f"Gold LOC entities: {gold_locs}")
        print(f"Baseline LOC entities: {b_locs}")
        print(f"Fixed LOC entities: {f_locs}")

        # Baseline errors
        b_fps = b_locs - gold_locs
        b_fns = gold_locs - b_locs
        # Fixed errors
        f_fps = f_locs - gold_locs
        f_fns = gold_locs - f_locs

        new_fps = f_fps - b_fps
        recovered_fps = b_fps - f_fps
        new_fns = f_fns - b_fns
        recovered_fns = b_fns - f_fns

        print(f"  Change: New FP = {new_fps} | Recovered FP = {recovered_fps} | New FN = {new_fns} | Recovered FN = {recovered_fns}")

        # Detailed token breakdown for differing spans
        for (etype, start, end) in (new_fps | new_fns | recovered_fps | recovered_fns):
            span_tokens = tokens[start:end]
            span_text = " ".join(span_tokens)
            has_conj = any(decompose_sinhala_conjuncts(t) != t for t in span_tokens)
            
            # Check presence in augmented dataset
            aug_info = []
            for t in span_tokens:
                occ = aug_tokens.get(t, [])
                dec_t = decompose_sinhala_conjuncts(t)
                dec_occ = aug_tokens.get(dec_t, [])
                aug_info.append({
                    "token": t,
                    "decomposed": dec_t,
                    "occurrences_in_aug": len(occ),
                    "tags_in_aug": set(o["tag"] for o in occ),
                    "dec_occurrences_in_aug": len(dec_occ),
                    "dec_tags_in_aug": set(o["tag"] for o in dec_occ),
                })

            print(f"  -> Span [{start}:{end}] '{span_text}':")
            print(f"     Gold: {case['gold'][start:end]}")
            print(f"     Baseline: {case['b_preds'][start:end]}")
            print(f"     Fixed: {case['f_preds'][start:end]}")
            print(f"     Contains Sinhala Conjunct: {has_conj}")
            for ai in aug_info:
                print(f"     Token '{ai['token']}': in conjunct_robustness.conll: {ai['occurrences_in_aug']} times {ai['tags_in_aug']} (dec '{ai['decomposed']}': {ai['dec_occurrences_in_aug']} times {ai['dec_tags_in_aug']})")

    # Also list ALL baseline FPs and Fixed FPs across whole dataset to understand what the 2 baseline FPs and 5 fixed FPs are!
    print("\n" + "=" * 90)
    print("GLOBAL LOC FALSE POSITIVES AUDIT")
    print("=" * 90)
    print("Baseline FPs (total 2):")
    for s_idx, b_rec in enumerate(base_records):
        golds = b_rec["gold"]
        b_preds = b_rec["pred"]
        tokens = b_rec["tokens"]
        gold_locs = {e for e in extract_entities(golds) if e[0] == "LOC"}
        b_locs = {e for e in extract_entities(b_preds) if e[0] == "LOC"}
        for fp in (b_locs - gold_locs):
            print(f"  Sent #{s_idx + 1}: span {fp} = '{' '.join(tokens[fp[1]:fp[2]])}' (Gold: {golds[fp[1]:fp[2]]}) Context: {' '.join(tokens[max(0, fp[1]-3):min(len(tokens), fp[2]+3)])}")

    print("\nConjunct-Fixed FPs (total 5):")
    for s_idx, f_rec in enumerate(fixed_records):
        golds = f_rec["gold"]
        f_preds = f_rec["pred"]
        tokens = f_rec["tokens"]
        gold_locs = {e for e in extract_entities(golds) if e[0] == "LOC"}
        f_locs = {e for e in extract_entities(f_preds) if e[0] == "LOC"}
        for fp in (f_locs - gold_locs):
            print(f"  Sent #{s_idx + 1}: span {fp} = '{' '.join(tokens[fp[1]:fp[2]])}' (Gold: {golds[fp[1]:fp[2]]}) Context: {' '.join(tokens[max(0, fp[1]-3):min(len(tokens), fp[2]+3)])}")


if __name__ == "__main__":
    analyze_loc_diff()
