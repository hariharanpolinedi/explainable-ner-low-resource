"""
Training pipeline for Word+Char-CNN BiLSTM-CRF NER models (Phase 6).

Features:
- Character and word joint dataset loader with script-aware grapheme tokenization
- Dynamic batch padding for both token sequences and character matrices
- Identical data condition, optimizer, learning rate, and early stopping as Phase 5
- Checkpoint persistence to checkpoints/charcnn_{lang}.pt
"""

from __future__ import annotations

import json
import os
import random
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
    Dataset = object  # type: ignore

from evaluation.seqeval_metrics import compute_seqeval_metrics
from model.ner_model_v2_charcnn import NERCharCNNModel
from preprocessing.tokenizer import ScriptTokenizer
from utils.config_loader import BaseConfig, load_config
from utils.seed import set_global_seed

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


class NERCharCoNLLDataset(Dataset):
    """
    Dataset for CoNLL NER data with dual word-token and grapheme-character indexing.
    """

    def __init__(
        self,
        conll_paths: list[Path | str],
        word_to_id: dict[str, int],
        char_to_id: dict[str, int],
        lang: str,
        pad_word_id: int = 0,
        unk_word_id: int = 1,
        pad_char_id: int = 0,
        unk_char_id: int = 1,
    ) -> None:
        self.word_to_id = word_to_id
        self.char_to_id = char_to_id
        self.lang = lang
        self.pad_word_id = pad_word_id
        self.unk_word_id = unk_word_id
        self.pad_char_id = pad_char_id
        self.unk_char_id = unk_char_id
        self.tokenizer = ScriptTokenizer(lang)
        self.samples: list[tuple[list[str], list[int], list[list[int]], list[int]]] = []
        self._load_files(conll_paths)

    def _load_files(self, paths: list[Path | str]) -> None:
        for p in paths:
            fpath = Path(p)
            if not fpath.is_file():
                raise FileNotFoundError(f"Missing CoNLL file: {fpath}")

            curr_tokens: list[str] = []
            curr_tags: list[int] = []

            with open(fpath, "r", encoding="utf-8") as fp:
                for line in fp:
                    line_str = line.strip()
                    if not line_str:
                        if curr_tokens:
                            self._append_sample(curr_tokens, curr_tags)
                            curr_tokens = []
                            curr_tags = []
                        continue

                    parts = line_str.split("\t")
                    if len(parts) == 2:
                        tok, tag = parts
                        curr_tokens.append(tok)
                        curr_tags.append(TAG_TO_ID.get(tag, 0))

            if curr_tokens:
                self._append_sample(curr_tokens, curr_tags)

    def _append_sample(self, tokens: list[str], tags: list[int]) -> None:
        word_ids = [self.word_to_id.get(t, self.unk_word_id) for t in tokens]
        char_ids_per_word: list[list[int]] = []
        for t in tokens:
            graphemes = self.tokenizer.tokenize_characters(t)
            c_ids = [self.char_to_id.get(g, self.unk_char_id) for g in graphemes]
            if not c_ids:
                c_ids = [self.unk_char_id]
            char_ids_per_word.append(c_ids)

        self.samples.append((tokens, word_ids, char_ids_per_word, tags))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[str], list[int], list[list[int]], list[int]]:
        return self.samples[idx]


def collate_fn_charcnn(
    batch: list[tuple[list[str], list[int], list[list[int]], list[int]]],
    min_word_len: int = 5,
) -> dict[str, Any]:
    """
    Collate and dynamically pad both token sequences and character matrices.
    """
    batch_tokens = [item[0] for item in batch]
    batch_word_ids = [item[1] for item in batch]
    batch_char_ids = [item[2] for item in batch]
    batch_tag_ids = [item[3] for item in batch]

    batch_size = len(batch)
    seq_lengths = [len(seq) for seq in batch_word_ids]
    max_seq_len = max(seq_lengths) if seq_lengths else 1

    # Find max word length across all words in this batch
    max_char_len = min_word_len
    for sent in batch_char_ids:
        for word_chars in sent:
            if len(word_chars) > max_char_len:
                max_char_len = len(word_chars)

    padded_word_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    padded_char_ids = torch.zeros((batch_size, max_seq_len, max_char_len), dtype=torch.long)
    padded_tags = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    for i in range(batch_size):
        slen = len(batch_word_ids[i])
        padded_word_ids[i, :slen] = torch.tensor(batch_word_ids[i], dtype=torch.long)
        padded_tags[i, :slen] = torch.tensor(batch_tag_ids[i], dtype=torch.long)
        mask[i, :slen] = True

        for j in range(slen):
            chars = batch_char_ids[i][j]
            clen = len(chars)
            padded_char_ids[i, j, :clen] = torch.tensor(chars, dtype=torch.long)

    return {
        "raw_tokens": batch_tokens,
        "word_ids": padded_word_ids,
        "char_ids": padded_char_ids,
        "tags": padded_tags,
        "mask": mask,
        "seq_lengths": seq_lengths,
    }


def evaluate_charcnn_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate model on a dev dataset using seqeval entity F1 and token accuracy."""
    model.eval()
    all_gold_tags: list[list[str]] = []
    all_pred_tags: list[list[str]] = []

    with torch.no_grad():
        for batch in dataloader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)
            seq_lengths = batch["seq_lengths"]

            best_paths = model(word_ids, char_ids, mask=mask)

            for i in range(len(seq_lengths)):
                slen = seq_lengths[i]
                gold_seq = [ID_TO_TAG[tid.item()] for tid in tags[i, :slen]]
                pred_seq = [ID_TO_TAG[tid] for tid in best_paths[i][:slen]]
                all_gold_tags.append(gold_seq)
                all_pred_tags.append(pred_seq)

    return compute_seqeval_metrics(all_gold_tags, all_pred_tags)


def train_charcnn_experiment(
    lang: str,
    train_paths: list[Path | str],
    dev_path: Path | str,
    word_vocab_path: Path | str,
    char_vocab_path: Path | str,
    checkpoint_out: Path | str,
    filter_widths: Sequence[int] = (3, 4, 5),
    filters_per_width: int = 30,
    bilstm_layers: int = 2,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 5,
    device_name: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train a Char-CNN BiLSTM-CRF model for a single language experiment.
    """
    if torch is None:
        raise ImportError("PyTorch must be installed for training.")

    set_global_seed(seed)

    # Determine compute device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    actual_device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    # Load vocabularies
    with open(word_vocab_path, "r", encoding="utf-8") as fp:
        word_vocab = json.load(fp)
    word_to_id = word_vocab["token_to_id"]
    word_vocab_size = word_vocab.get("vocab_size", len(word_to_id))

    with open(char_vocab_path, "r", encoding="utf-8") as fp:
        char_vocab = json.load(fp)
    char_to_id = char_vocab["token_to_id"]
    char_vocab_size = char_vocab.get("vocab_size", len(char_to_id))

    min_wlen = max(filter_widths) if filter_widths else 5

    # Build datasets
    train_dataset = NERCharCoNLLDataset(
        conll_paths=train_paths,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
        lang=lang,
    )
    dev_dataset = NERCharCoNLLDataset(
        conll_paths=[dev_path],
        word_to_id=word_to_id,
        char_to_id=char_to_id,
        lang=lang,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn_charcnn(b, min_word_len=min_wlen),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn_charcnn(b, min_word_len=min_wlen),
    )

    # Instantiate model
    model = NERCharCNNModel(
        word_vocab_size=word_vocab_size,
        char_vocab_size=char_vocab_size,
        num_tags=len(TAG_TO_ID),
        word_embedding_dim=100,
        char_embedding_dim=50,
        filter_widths=filter_widths,
        filters_per_width=filters_per_width,
        hidden_dim=128,
        num_layers=bilstm_layers,
        word_dropout=0.3,
        char_dropout=0.3,
        concat_dropout=0.3,
        bilstm_dropout=0.3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_dev_entity_f1 = -1.0
    best_epoch = -1
    best_metrics: dict[str, Any] = {}
    patience_counter = 0

    checkpoint_path = Path(checkpoint_out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            loss = model(word_ids, char_ids, tags=tags, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Evaluate on gold-dev
        dev_metrics = evaluate_charcnn_model(model, dev_loader, device)
        dev_f1 = dev_metrics["overall"]["entity_f1"]
        dev_acc = dev_metrics["overall"]["token_accuracy"]

        # Check early stopping
        if dev_f1 > best_dev_entity_f1:
            best_dev_entity_f1 = dev_f1
            best_epoch = epoch
            best_metrics = dev_metrics
            patience_counter = 0

            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_dev_f1": best_dev_entity_f1,
                    "best_dev_acc": dev_acc,
                    "metrics": dev_metrics,
                    "word_vocab_size": word_vocab_size,
                    "char_vocab_size": char_vocab_size,
                    "filter_widths": tuple(filter_widths),
                    "filters_per_width": filters_per_width,
                    "num_layers": bilstm_layers,
                    "language": lang,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    duration = time.time() - start_time

    return {
        "language": lang,
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "best_dev_entity_f1": best_dev_entity_f1,
        "best_dev_token_acc": best_metrics.get("overall", {}).get("token_accuracy", 0.0),
        "best_metrics": best_metrics,
        "duration_seconds": round(duration, 2),
        "device_name": actual_device_name,
        "checkpoint_path": str(checkpoint_path),
        "filter_widths": list(filter_widths),
        "filters_per_width": filters_per_width,
    }
