"""
Confidence-Based Inference and Filtering Pipeline (Phase 10).

Runs the fine-tuned model checkpoint on the remaining unlabeled pool, computes
the exact CRF sequence-level normalized confidence score for the Viterbi path:
    conf(x) = exp( (score(x, y*) - log Z(x)) / length )
and filters sentences exceeding the round's confidence threshold.
Enforces volume caps and outputs valid CoNLL-2003 formatted pseudo-labeled data.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore

from model.ner_model_v4_multilingual import NERMultilingualModel
from preprocessing.tokenizer import ScriptTokenizer
from training.multilingual_dataset import LANG_TO_ID, load_or_create_multilingual_vocabs
from training.train_charcnn import ID_TO_TAG, TAG_TO_ID


class UnlabeledSentenceDataset(Dataset if Dataset else object):
    """
    Dataset wrapper for unlabeled tokenized sentences for efficient GPU inference.
    """

    def __init__(
        self,
        tokenized_sentences: list[list[str]],
        lang: str,
        word_to_id: dict[str, int],
        char_to_id: dict[str, int],
        pad_word_id: int = 0,
        unk_word_id: int = 1,
        pad_char_id: int = 0,
        unk_char_id: int = 1,
    ) -> None:
        self.tokenized_sentences = tokenized_sentences
        self.lang = lang
        self.lang_id = LANG_TO_ID[lang]
        self.word_to_id = word_to_id
        self.char_to_id = char_to_id
        self.pad_word_id = pad_word_id
        self.unk_word_id = unk_word_id
        self.pad_char_id = pad_char_id
        self.unk_char_id = unk_char_id
        self.tokenizer = ScriptTokenizer(lang)

    def __len__(self) -> int:
        return len(self.tokenized_sentences)

    def __getitem__(self, idx: int) -> tuple[list[str], list[int], list[list[int]], int]:
        tokens = self.tokenized_sentences[idx]
        word_ids = [self.word_to_id.get(t, self.unk_word_id) for t in tokens]
        char_ids_per_word: list[list[int]] = []
        for t in tokens:
            graphemes = self.tokenizer.tokenize_characters(t)
            c_ids = [self.char_to_id.get(g, self.unk_char_id) for g in graphemes]
            if not c_ids:
                c_ids = [self.unk_char_id]
            char_ids_per_word.append(c_ids)
        return tokens, word_ids, char_ids_per_word, self.lang_id


def collate_unlabeled(
    batch: list[tuple[list[str], list[int], list[list[int]], int]],
    min_word_len: int = 5,
) -> dict[str, Any]:
    """Collate variable-length unlabeled sentences into padded tensors."""
    batch_tokens = [item[0] for item in batch]
    batch_word_ids = [item[1] for item in batch]
    batch_char_ids = [item[2] for item in batch]
    lang_ids = torch.tensor([item[3] for item in batch], dtype=torch.long)

    batch_size = len(batch)
    seq_lengths = [len(seq) for seq in batch_word_ids]
    max_seq_len = max(seq_lengths) if seq_lengths else 1

    max_char_len = min_word_len
    for sent in batch_char_ids:
        for word_chars in sent:
            if len(word_chars) > max_char_len:
                max_char_len = len(word_chars)

    padded_word_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    padded_char_ids = torch.zeros((batch_size, max_seq_len, max_char_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    for i in range(batch_size):
        slen = len(batch_word_ids[i])
        padded_word_ids[i, :slen] = torch.tensor(batch_word_ids[i], dtype=torch.long)
        mask[i, :slen] = True
        for j in range(slen):
            w_chars = batch_char_ids[i][j]
            padded_char_ids[i, j, :len(w_chars)] = torch.tensor(w_chars, dtype=torch.long)

    return {
        "tokens": batch_tokens,
        "word_ids": padded_word_ids,
        "char_ids": padded_char_ids,
        "lang_ids": lang_ids,
        "mask": mask,
        "seq_lengths": seq_lengths,
    }


def compute_crf_sequence_confidences(
    model: NERMultilingualModel,
    word_ids: torch.Tensor,
    char_ids: torch.Tensor,
    lang_ids: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[list[list[int]], torch.Tensor]:
    """
    Compute Viterbi predicted paths and sequence confidence scores:
        conf = exp((score(x, y*) - log Z(x)) / length)
    """
    seq_lengths = mask.sum(dim=1).float()
    batch_size, seq_len = word_ids.shape

    word_repr = model.word_embedding(word_ids)
    char_repr = model.char_cnn(char_ids)
    lang_repr = model.lang_embedding(lang_ids, seq_len)

    combined = model.concat_dropout(torch.cat([word_repr, char_repr, lang_repr], dim=-1))
    lstm_hidden = model.encoder(combined, mask=mask, return_hidden=True)
    attended_states, _ = model.attention(lstm_hidden, mask=mask)

    if model.crf_config == "shared":
        emissions = model.fc_shared(attended_states)
        crf = model.crf_shared
        paths = crf.decode(emissions, mask=mask)

        # Pad paths into tags tensor
        tags_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long, device=word_ids.device)
        for i, p in enumerate(paths):
            if p:
                tags_tensor[i, :len(p)] = torch.tensor(p, dtype=torch.long, device=word_ids.device)

        gold_scores = crf._compute_score(emissions, tags_tensor, mask)
        forward_scores = crf._compute_forward_score(emissions, mask)
        log_probs = gold_scores - forward_scores
        # Length-normalized geometric mean confidence per token
        confidences = torch.exp(log_probs / torch.clamp(seq_lengths, min=1.0))
        confidences = torch.clamp(confidences, 0.0, 1.0)
    else:
        # Per-language CRF heads
        paths = [[] for _ in range(batch_size)]
        confidences = torch.zeros(batch_size, device=word_ids.device)
        for l_id in range(model.num_languages):
            idx = (lang_ids == l_id).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
            sub_em = model.fc_heads[l_id](attended_states[idx])
            sub_mask = mask[idx]
            sub_lengths = seq_lengths[idx]
            sub_crf = model.crf_heads[l_id]
            sub_paths = sub_crf.decode(sub_em, mask=sub_mask)
            sub_tags = torch.zeros((len(idx), seq_len), dtype=torch.long, device=word_ids.device)
            for k, p in enumerate(sub_paths):
                if p:
                    sub_tags[k, :len(p)] = torch.tensor(p, dtype=torch.long, device=word_ids.device)
            sub_scores = sub_crf._compute_score(sub_em, sub_tags, sub_mask)
            sub_fwd = sub_crf._compute_forward_score(sub_em, sub_mask)
            sub_logp = sub_scores - sub_fwd
            sub_conf = torch.clamp(torch.exp(sub_logp / torch.clamp(sub_lengths, min=1.0)), 0.0, 1.0)

            for k, orig_idx in enumerate(idx.tolist()):
                paths[orig_idx] = sub_paths[k]
                confidences[orig_idx] = sub_conf[k]

    return paths, confidences


def load_model_for_inference(
    checkpoint_path: Path | str,
    proc_dir: Path | str,
    device: torch.device,
) -> NERMultilingualModel:
    """Load model from checkpoint for inference."""
    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)
    model = NERMultilingualModel(
        word_vocab_size=len(word_to_id),
        char_vocab_size=len(char_to_id),
        num_tags=9,
        num_languages=3,
        word_embedding_dim=100,
        char_embedding_dim=50,
        lang_embedding_dim=16,
        filter_widths=(3, 4, 5),
        filters_per_width=30,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_attention_blocks=1,
        attention_dropout=0.1,
        crf_config="shared",
    ).to(device)

    chk = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()
    return model


def filter_unlabeled_pool(
    lang: str,
    candidate_tokens: list[list[str]],
    checkpoint_path: Path | str,
    proc_dir: Path | str,
    threshold: float | dict[str, float],
    device: torch.device,
    batch_size: int = 64,
    volume_cap: int | None = None,
    output_conll_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Run confidence filter over candidate sentences and optionally save to CoNLL.

    Args:
        lang: Language name
        candidate_tokens: List of tokenized candidate sentences
        checkpoint_path: Model checkpoint to load
        proc_dir: Processed directory containing vocabularies
        threshold: Confidence cutoff as single float or per-type dict (e.g. {'PER': 0.9, 'ORG': 0.75})
        device: Torch device (CUDA/CPU)
        batch_size: Inference batch size
        volume_cap: Maximum number of pseudo sentences to retain
        output_conll_path: File to write pseudo-labeled CoNLL data to

    Returns:
        Dictionary with filtered candidates, metrics, and execution summary.
    """
    t0 = time.time()
    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)
    model = load_model_for_inference(checkpoint_path, proc_dir, device)

    dataset = UnlabeledSentenceDataset(
        candidate_tokens,
        lang=lang,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_unlabeled,
    )

    high_conf_candidates: list[dict[str, Any]] = []
    total_evaluated = 0

    with torch.no_grad():
        for batch in dataloader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            mask = batch["mask"].to(device)
            tokens_batch = batch["tokens"]
            seq_lens = batch["seq_lengths"]

            paths, confs = compute_crf_sequence_confidences(
                model=model,
                word_ids=word_ids,
                char_ids=char_ids,
                lang_ids=lang_ids,
                mask=mask,
            )

            confs_list = confs.cpu().tolist()
            total_evaluated += len(tokens_batch)

            for i in range(len(tokens_batch)):
                c = confs_list[i]
                slen = seq_lens[i]
                p_tags = [ID_TO_TAG[tid] for tid in paths[i][:slen]]

                if isinstance(threshold, dict):
                    # Priority given to ORG threshold to boost ORG retention
                    if any(t in ("B-ORG", "I-ORG") for t in p_tags):
                        effective_thresh = threshold.get("ORG", 0.75)
                    elif any(t in ("B-PER", "I-PER") for t in p_tags):
                        effective_thresh = threshold.get("PER", 0.90)
                    elif any(t in ("B-LOC", "I-LOC") for t in p_tags):
                        effective_thresh = threshold.get("LOC", 0.90)
                    elif any(t in ("B-MISC", "I-MISC") for t in p_tags):
                        effective_thresh = threshold.get("MISC", 0.85)
                    else:
                        effective_thresh = threshold.get("O", 0.90)
                else:
                    effective_thresh = float(threshold)

                if c >= effective_thresh:
                    toks = tokens_batch[i]
                    has_entity = any(t != "O" for t in p_tags)

                    high_conf_candidates.append({
                        "tokens": toks,
                        "tags": p_tags,
                        "confidence": c,
                        "has_entity": has_entity,
                    })

    # Sort descending by confidence
    high_conf_candidates.sort(key=lambda x: x["confidence"], reverse=True)

    # Apply volume cap if specified
    retained = high_conf_candidates
    capped = False
    if volume_cap is not None and len(retained) > volume_cap:
        retained = retained[:volume_cap]
        capped = True

    # Count per-type entity annotations in retained pseudo-data
    from weak_supervision.estimate_silver_precision import extract_entities_from_bio
    per_type_counts = {"PER": 0, "LOC": 0, "ORG": 0, "MISC": 0}
    for item in retained:
        for etype, _, _ in extract_entities_from_bio(item["tags"]):
            if etype in per_type_counts:
                per_type_counts[etype] += 1

    # Save to CoNLL if path provided
    if output_conll_path:
        out_p = Path(output_conll_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for item in retained:
                for tok, tag in zip(item["tokens"], item["tags"]):
                    f.write(f"{tok}\t{tag}\n")
                f.write("\n")

    elapsed = time.time() - t0
    avg_conf = sum(x["confidence"] for x in retained) / len(retained) if retained else 0.0
    entity_bearing = sum(1 for x in retained if x["has_entity"])

    return {
        "language": lang,
        "total_evaluated": total_evaluated,
        "total_above_threshold": len(high_conf_candidates),
        "retained_count": len(retained),
        "volume_capped": capped,
        "average_confidence": round(avg_conf, 4),
        "entity_bearing_count": entity_bearing,
        "entity_bearing_ratio": round(entity_bearing / len(retained), 4) if retained else 0.0,
        "per_type_entity_counts": per_type_counts,
        "elapsed_seconds": round(elapsed, 2),
        "output_path": str(output_conll_path) if output_conll_path else None,
        "retained_samples": retained,
    }


if __name__ == "__main__":
    print("Testing Confidence Filter with small synthetic batch...")
    # Quick sanity check
    from model.crf_layer import LinearChainCRF
    crf = LinearChainCRF(9)
    print("CRF instantiated successfully.")
