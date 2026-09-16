"""
Out-Of-Vocabulary (OOV) vs In-Vocabulary (IV) performance analysis module (Phase 6).

Evaluates and compares Phase 5 (Baseline) vs Phase 6 (Char-CNN) checkpoints on gold-dev:
- Splits tokens and entities into In-Vocabulary (IV) and Out-Of-Vocabulary (OOV) subsets
  based on reference vocabulary (word_vocab.json or training corpus).
- Computes entity-level precision, recall, F1 (exact span match) on:
  1. Overall
  2. In-Vocabulary (IV) subset
  3. Out-of-Vocabulary (OOV) subset
- Computes token-level accuracy on IV vs OOV tokens.
- Calculates delta: Delta = Phase_6 - Phase_5.
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

from evaluation.seqeval_metrics import compute_seqeval_metrics, extract_entities
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


def evaluate_oov_iv_split(
    sentences: list[list[str]],
    gold_tags: list[list[str]],
    pred_tags: list[list[str]],
    in_vocab_words: set[str],
) -> dict[str, Any]:
    """
    Split entities and tokens into In-Vocabulary (IV) and Out-of-Vocabulary (OOV) subsets
    and compute evaluation metrics on each subset.
    """
    # 1. Overall metrics via standard seqeval
    overall_metrics = compute_seqeval_metrics(gold_tags, pred_tags)

    # 2. Span-level IV vs OOV evaluation
    gold_iv_spans: list[tuple[str, int, int]] = []
    gold_oov_spans: list[tuple[str, int, int]] = []
    pred_iv_spans: list[tuple[str, int, int]] = []
    pred_oov_spans: list[tuple[str, int, int]] = []

    total_iv_tokens = 0
    correct_iv_tokens = 0
    total_oov_tokens = 0
    correct_oov_tokens = 0

    for sent_idx, (words, g_tags, p_tags) in enumerate(zip(sentences, gold_tags, pred_tags)):
        # Check token-level accuracy per word
        for w, gt, pt in zip(words, g_tags, p_tags):
            is_oov = w not in in_vocab_words
            if is_oov:
                total_oov_tokens += 1
                if gt == pt:
                    correct_oov_tokens += 1
            else:
                total_iv_tokens += 1
                if gt == pt:
                    correct_iv_tokens += 1

        # Extract entity spans for this sentence
        g_entities = extract_entities(g_tags)
        p_entities = extract_entities(p_tags)

        # Categorize gold entities
        for ent_type, start, end in g_entities:
            span_words = words[start : end + 1]
            span_is_oov = any(sw not in in_vocab_words for sw in span_words)
            span_key = (f"{sent_idx}_{ent_type}", start, end)
            if span_is_oov:
                gold_oov_spans.append(span_key)
            else:
                gold_iv_spans.append(span_key)

        # Categorize pred entities
        for ent_type, start, end in p_entities:
            span_words = words[start : end + 1]
            span_is_oov = any(sw not in in_vocab_words for sw in span_words)
            span_key = (f"{sent_idx}_{ent_type}", start, end)
            if span_is_oov:
                pred_oov_spans.append(span_key)
            else:
                pred_iv_spans.append(span_key)

    # Calculate IV metrics
    set_gold_iv = set(gold_iv_spans)
    set_pred_iv = set(pred_iv_spans)
    correct_iv = len(set_gold_iv.intersection(set_pred_iv))
    prec_iv = correct_iv / len(set_pred_iv) if set_pred_iv else (1.0 if not set_gold_iv else 0.0)
    rec_iv = correct_iv / len(set_gold_iv) if set_gold_iv else 0.0
    f1_iv = (2 * prec_iv * rec_iv / (prec_iv + rec_iv)) if (prec_iv + rec_iv) > 0 else 0.0
    acc_iv = correct_iv_tokens / total_iv_tokens if total_iv_tokens > 0 else 0.0

    # Calculate OOV metrics
    set_gold_oov = set(gold_oov_spans)
    set_pred_oov = set(pred_oov_spans)
    correct_oov = len(set_gold_oov.intersection(set_pred_oov))
    prec_oov = correct_oov / len(set_pred_oov) if set_pred_oov else (1.0 if not set_gold_oov else 0.0)
    rec_oov = correct_oov / len(set_gold_oov) if set_gold_oov else 0.0
    f1_oov = (2 * prec_oov * rec_oov / (prec_oov + rec_oov)) if (prec_oov + rec_oov) > 0 else 0.0
    acc_oov = correct_oov_tokens / total_oov_tokens if total_oov_tokens > 0 else 0.0

    return {
        "overall": {
            "entity_precision": overall_metrics["overall"]["entity_precision"],
            "entity_recall": overall_metrics["overall"]["entity_recall"],
            "entity_f1": overall_metrics["overall"]["entity_f1"],
            "token_accuracy": overall_metrics["overall"]["token_accuracy"],
            "total_entities": len(extract_entities([tag for seq in gold_tags for tag in seq])),
        },
        "in_vocab": {
            "entity_precision": prec_iv,
            "entity_recall": rec_iv,
            "entity_f1": f1_iv,
            "token_accuracy": acc_iv,
            "gold_entities_count": len(set_gold_iv),
            "pred_entities_count": len(set_pred_iv),
            "total_tokens": total_iv_tokens,
        },
        "out_of_vocab": {
            "entity_precision": prec_oov,
            "entity_recall": rec_oov,
            "entity_f1": f1_oov,
            "token_accuracy": acc_oov,
            "gold_entities_count": len(set_gold_oov),
            "pred_entities_count": len(set_pred_oov),
            "total_tokens": total_oov_tokens,
        },
    }


def predict_dataset_baseline(
    model: Any,
    samples: list[tuple[list[str], list[str]]],
    word_to_id: dict[str, int],
    device: torch.device,
) -> list[list[str]]:
    """Generate predictions using Phase 5 baseline model."""
    model.eval()
    all_preds: list[list[str]] = []

    with torch.no_grad():
        for words, _ in samples:
            token_ids = [word_to_id.get(w, 1) for w in words]
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
            mask = torch.ones_like(input_tensor, dtype=torch.bool)
            paths = model(input_tensor, mask=mask)
            pred_tags = [ID_TO_TAG.get(tag_id, "O") for tag_id in paths[0]]
            all_preds.append(pred_tags)

    return all_preds


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


def run_comparative_oov_analysis(
    lang: str,
    baseline_checkpoint_path: Path | str,
    charcnn_checkpoint_path: Path | str,
    dev_path: Path | str,
    word_vocab_path: Path | str,
    char_vocab_path: Path | str,
    device: torch.device,
    vocab_reference_type: str = "word_vocab",  # 'word_vocab' or 'training_data'
    train_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Run comparative OOV vs IV evaluation between Phase 5 and Phase 6 checkpoints.
    """
    if torch is None:
        raise ImportError("PyTorch required for OOV analysis.")

    from model.ner_model_v1_baseline import NERBaselineModel
    from model.ner_model_v2_charcnn import NERCharCNNModel

    # 1. Load vocabularies
    with open(word_vocab_path, "r", encoding="utf-8") as fp:
        word_vocab = json.load(fp)
    word_to_id = word_vocab["token_to_id"]
    word_vocab_size = word_vocab.get("vocab_size", len(word_to_id))

    with open(char_vocab_path, "r", encoding="utf-8") as fp:
        char_vocab = json.load(fp)
    char_to_id = char_vocab["token_to_id"]
    char_vocab_size = char_vocab.get("vocab_size", len(char_to_id))

    # Determine reference vocabulary for IV vs OOV distinction
    if vocab_reference_type == "training_data" and train_path is not None:
        in_vocab_words: set[str] = set()
        with open(train_path, "r", encoding="utf-8") as fp:
            for line in fp:
                line_str = line.strip()
                if line_str:
                    parts = line_str.split("\t")
                    if len(parts) == 2:
                        in_vocab_words.add(parts[0])
    else:
        # Default: in-vocabulary words from word_vocab.json
        in_vocab_words = set(word_to_id.keys())
        in_vocab_words.discard("<PAD>")
        in_vocab_words.discard("<UNK>")

    # 2. Load dev data
    dev_samples = load_conll_data(dev_path)
    sentences = [s[0] for s in dev_samples]
    gold_tags = [s[1] for s in dev_samples]

    tokenizer = ScriptTokenizer(lang)

    # 3. Load Phase 5 Baseline Model
    chk_p5 = torch.load(baseline_checkpoint_path, map_location=device, weights_only=False)
    p5_layers = chk_p5.get("num_layers", 2)
    model_p5 = NERBaselineModel(
        vocab_size=word_vocab_size,
        num_tags=len(TAG_TO_ID),
        embedding_dim=100,
        hidden_dim=128,
        num_layers=p5_layers,
    ).to(device)
    model_p5.load_state_dict(chk_p5["model_state_dict"])

    preds_p5 = predict_dataset_baseline(model_p5, dev_samples, word_to_id, device)
    metrics_p5 = evaluate_oov_iv_split(sentences, gold_tags, preds_p5, in_vocab_words)

    # 4. Load Phase 6 Char-CNN Model
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
    metrics_p6 = evaluate_oov_iv_split(sentences, gold_tags, preds_p6, in_vocab_words)

    # 5. Compute Deltas (Phase 6 - Phase 5)
    delta_overall_f1 = metrics_p6["overall"]["entity_f1"] - metrics_p5["overall"]["entity_f1"]
    delta_iv_f1 = metrics_p6["in_vocab"]["entity_f1"] - metrics_p5["in_vocab"]["entity_f1"]
    delta_oov_f1 = metrics_p6["out_of_vocab"]["entity_f1"] - metrics_p5["out_of_vocab"]["entity_f1"]

    delta_overall_acc = metrics_p6["overall"]["token_accuracy"] - metrics_p5["overall"]["token_accuracy"]
    delta_iv_acc = metrics_p6["in_vocab"]["token_accuracy"] - metrics_p5["in_vocab"]["token_accuracy"]
    delta_oov_acc = metrics_p6["out_of_vocab"]["token_accuracy"] - metrics_p5["out_of_vocab"]["token_accuracy"]

    return {
        "language": lang,
        "vocab_reference_type": vocab_reference_type,
        "phase5_baseline": metrics_p5,
        "phase6_charcnn": metrics_p6,
        "deltas": {
            "overall_entity_f1": delta_overall_f1,
            "in_vocab_entity_f1": delta_iv_f1,
            "out_of_vocab_entity_f1": delta_oov_f1,
            "overall_token_acc": delta_overall_acc,
            "in_vocab_token_acc": delta_iv_acc,
            "out_of_vocab_token_acc": delta_oov_acc,
        },
    }
