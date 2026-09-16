"""
Span-length-stratified entity F1 evaluation module (Phase 7).

Evaluates and compares Phase 6 (Char-CNN) vs Phase 7 (Self-Attention) checkpoints on gold-dev:
- Stratifies entities into:
  1. Single-token entities (length == 1)
  2. Multi-token entities (length >= 2)
- Computes exact-span precision, recall, and F1 per stratum.
- Computes Deltas (Phase 7 - Phase 6) to verify hypothesis that self-attention
  provides pronounced gains on multi-token contextual spans.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from evaluation.seqeval_metrics import extract_entities
from preprocessing.tokenizer import ScriptTokenizer

TAG_TO_ID = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}
ID_TO_TAG = {v: k for k, v in TAG_TO_ID.items()}


def load_conll_data(conll_path: Path | str) -> list[tuple[list[str], list[str]]]:
    """Load sentences and corresponding gold BIO tags from a CoNLL file."""
    samples: list[tuple[list[str], list[str]]] = []
    curr_tokens: list[str] = []
    curr_tags: list[str] = []

    with open(conll_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line_str = line.strip()
            if not line_str:
                if curr_tokens:
                    samples.append((curr_tokens, curr_tags))
                    curr_tokens = []
                    curr_tags = []
                continue

            parts = line_str.split("\t")
            if len(parts) == 2:
                tok, tag = parts
                curr_tokens.append(tok)
                curr_tags.append(tag)

    if curr_tokens:
        samples.append((curr_tokens, curr_tags))

    return samples


def evaluate_span_length_stratified(
    gold_tags: list[list[str]],
    pred_tags: list[list[str]],
) -> dict[str, Any]:
    """
    Stratify entities by token length (single-token: 1 vs multi-token: >=2)
    and compute exact-match precision, recall, and F1 for each stratum.
    """
    gold_single: list[tuple[str, int, int]] = []
    gold_multi: list[tuple[str, int, int]] = []
    pred_single: list[tuple[str, int, int]] = []
    pred_multi: list[tuple[str, int, int]] = []

    for sent_idx, (g_seq, p_seq) in enumerate(zip(gold_tags, pred_tags)):
        g_entities = extract_entities(g_seq)
        p_entities = extract_entities(p_seq)

        for ent_type, start, end in g_entities:
            span_len = end - start
            key = (f"{sent_idx}_{ent_type}", start, end)
            if span_len == 1:
                gold_single.append(key)
            else:
                gold_multi.append(key)

        for ent_type, start, end in p_entities:
            span_len = end - start
            key = (f"{sent_idx}_{ent_type}", start, end)
            if span_len == 1:
                pred_single.append(key)
            else:
                pred_multi.append(key)

    def _calc_metrics(gold_set: set, pred_set: set) -> dict[str, Any]:
        correct = len(gold_set.intersection(pred_set))
        prec = correct / len(pred_set) if pred_set else (1.0 if not gold_set else 0.0)
        rec = correct / len(gold_set) if gold_set else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "gold_count": len(gold_set),
            "pred_count": len(pred_set),
            "correct_count": correct,
        }

    single_metrics = _calc_metrics(set(gold_single), set(pred_single))
    multi_metrics = _calc_metrics(set(gold_multi), set(pred_multi))
    all_metrics = _calc_metrics(set(gold_single + gold_multi), set(pred_single + pred_multi))

    return {
        "overall": all_metrics,
        "single_token": single_metrics,
        "multi_token": multi_metrics,
    }


def predict_dataset_charcnn(
    model: Any,
    samples: list[tuple[list[str], list[str]]],
    word_to_id: dict[str, int],
    char_to_id: dict[str, int],
    tokenizer: ScriptTokenizer,
    device: torch.device,
) -> list[list[str]]:
    """Generate predictions using Phase 6 Char-CNN model."""
    model.eval()
    all_preds: list[list[str]] = []

    with torch.no_grad():
        for words, _ in samples:
            token_ids = [word_to_id.get(w, 1) for w in words]
            word_chars_list = [tokenizer.tokenize_characters(w) for w in words]
            max_word_len = max(max(len(c) for c in word_chars_list), 5)

            char_matrix: list[list[int]] = []
            for chars in word_chars_list:
                c_ids = [char_to_id.get(c, 1) for c in chars]
                pad_len = max_word_len - len(c_ids)
                char_matrix.append(c_ids + [0] * pad_len)

            word_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
            char_tensor = torch.tensor([char_matrix], dtype=torch.long, device=device)
            mask = torch.ones_like(word_tensor, dtype=torch.bool)

            paths = model(word_tensor, char_tensor, mask=mask)
            pred_tags = [ID_TO_TAG.get(tag_id, "O") for tag_id in paths[0]]
            all_preds.append(pred_tags)

    return all_preds


def predict_dataset_attention(
    model: Any,
    samples: list[tuple[list[str], list[str]]],
    word_to_id: dict[str, int],
    char_to_id: dict[str, int],
    tokenizer: ScriptTokenizer,
    device: torch.device,
) -> list[list[str]]:
    """Generate predictions using Phase 7 Attention model."""
    model.eval()
    all_preds: list[list[str]] = []

    with torch.no_grad():
        for words, _ in samples:
            token_ids = [word_to_id.get(w, 1) for w in words]
            word_chars_list = [tokenizer.tokenize_characters(w) for w in words]
            max_word_len = max(max(len(c) for c in word_chars_list), 5)

            char_matrix: list[list[int]] = []
            for chars in word_chars_list:
                c_ids = [char_to_id.get(c, 1) for c in chars]
                pad_len = max_word_len - len(c_ids)
                char_matrix.append(c_ids + [0] * pad_len)

            word_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
            char_tensor = torch.tensor([char_matrix], dtype=torch.long, device=device)
            mask = torch.ones_like(word_tensor, dtype=torch.bool)

            paths = model(word_tensor, char_tensor, mask=mask)
            pred_tags = [ID_TO_TAG.get(tag_id, "O") for tag_id in paths[0]]
            all_preds.append(pred_tags)

    return all_preds


def run_comparative_span_length_analysis(
    lang: str,
    charcnn_checkpoint_path: Path | str,
    attention_checkpoint_path: Path | str,
    dev_path: Path | str,
    word_vocab_path: Path | str,
    char_vocab_path: Path | str,
    device: torch.device,
) -> dict[str, Any]:
    """Run comparative span-length stratified analysis between Phase 6 and Phase 7 checkpoints."""
    if torch is None:
        raise ImportError("PyTorch required for span-length analysis.")

    from model.ner_model_v2_charcnn import NERCharCNNModel
    from model.ner_model_v3_attention import NERAttentionModel

    # 1. Load vocabularies
    with open(word_vocab_path, "r", encoding="utf-8") as fp:
        word_vocab = json.load(fp)
    word_to_id = word_vocab["token_to_id"]
    word_vocab_size = word_vocab.get("vocab_size", len(word_to_id))

    with open(char_vocab_path, "r", encoding="utf-8") as fp:
        char_vocab = json.load(fp)
    char_to_id = char_vocab["token_to_id"]
    char_vocab_size = char_vocab.get("vocab_size", len(char_to_id))

    # 2. Load dev data
    dev_samples = load_conll_data(dev_path)
    gold_tags = [s[1] for s in dev_samples]
    tokenizer = ScriptTokenizer(lang)

    # 3. Load Phase 6 Char-CNN Model
    chk_p6 = torch.load(charcnn_checkpoint_path, map_location=device, weights_only=False)
    p6_filter_widths = chk_p6.get("filter_widths", (3, 4, 5))
    p6_filters_per_width = chk_p6.get("filters_per_width", 30)
    p6_layers = chk_p6.get("num_layers", 2)

    model_p6 = NERCharCNNModel(
        word_vocab_size=word_vocab_size,
        char_vocab_size=char_vocab_size,
        num_tags=len(TAG_TO_ID),
        word_embedding_dim=100,
        char_embedding_dim=50,
        filter_widths=p6_filter_widths,
        filters_per_width=p6_filters_per_width,
        hidden_dim=128,
        num_layers=p6_layers,
    ).to(device)
    model_p6.load_state_dict(chk_p6["model_state_dict"])
    preds_p6 = predict_dataset_charcnn(model_p6, dev_samples, word_to_id, char_to_id, tokenizer, device)
    metrics_p6 = evaluate_span_length_stratified(gold_tags, preds_p6)

    # 4. Load Phase 7 Attention Model
    chk_p7 = torch.load(attention_checkpoint_path, map_location=device, weights_only=False)
    p7_filter_widths = chk_p7.get("filter_widths", (3, 4, 5))
    p7_filters_per_width = chk_p7.get("filters_per_width", 30)
    p7_layers = chk_p7.get("num_layers", 2)
    p7_heads = chk_p7.get("num_heads", 4)
    p7_blocks = chk_p7.get("num_attention_blocks", 1)

    model_p7 = NERAttentionModel(
        word_vocab_size=word_vocab_size,
        char_vocab_size=char_vocab_size,
        num_tags=len(TAG_TO_ID),
        word_embedding_dim=100,
        char_embedding_dim=50,
        filter_widths=p7_filter_widths,
        filters_per_width=p7_filters_per_width,
        hidden_dim=128,
        num_layers=p7_layers,
        num_heads=p7_heads,
        num_attention_blocks=p7_blocks,
    ).to(device)
    model_p7.load_state_dict(chk_p7["model_state_dict"])
    preds_p7 = predict_dataset_attention(model_p7, dev_samples, word_to_id, char_to_id, tokenizer, device)
    metrics_p7 = evaluate_span_length_stratified(gold_tags, preds_p7)

    # 5. Compute Deltas (Phase 7 - Phase 6)
    delta_single = metrics_p7["single_token"]["f1"] - metrics_p6["single_token"]["f1"]
    delta_multi = metrics_p7["multi_token"]["f1"] - metrics_p6["multi_token"]["f1"]
    delta_overall = metrics_p7["overall"]["f1"] - metrics_p6["overall"]["f1"]

    return {
        "language": lang,
        "phase6_charcnn": metrics_p6,
        "phase7_attention": metrics_p7,
        "deltas": {
            "overall_f1": delta_overall,
            "single_token_f1": delta_single,
            "multi_token_f1": delta_multi,
        },
    }
