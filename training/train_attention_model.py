"""
Training pipeline for Word+Char-CNN BiLSTM Self-Attention CRF NER models (Phase 7).

Features:
- Multi-Head Self-Attention integration over BiLSTM contextual states
- Linear learning rate warmup for attention stability
- Identical data condition, optimizer, learning rate, and early stopping as Phase 6
- Checkpoint persistence to checkpoints/attention_{lang}.pt
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
    from torch.utils.data import DataLoader
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore

from evaluation.seqeval_metrics import compute_seqeval_metrics
from model.ner_model_v3_attention import NERAttentionModel
from training.train_charcnn import NERCharCoNLLDataset, collate_fn_charcnn
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


def evaluate_attention_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate attention model on dev dataset using seqeval entity F1 and token accuracy."""
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


def train_attention_experiment(
    lang: str,
    train_paths: list[Path | str],
    dev_path: Path | str,
    word_vocab_path: Path | str,
    char_vocab_path: Path | str,
    checkpoint_out: Path | str,
    filter_widths: Sequence[int] = (3, 4, 5),
    filters_per_width: int = 30,
    bilstm_layers: int = 2,
    num_heads: int = 4,
    num_attention_blocks: int = 1,
    attention_dropout: float = 0.1,
    warmup_steps: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 5,
    device_name: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train a Multi-Head Self-Attention BiLSTM-CRF model for a single language experiment.
    """
    if torch is None:
        raise ImportError("PyTorch must be installed for training.")

    set_global_seed(seed)

    # Device selection
    if device_name in ["auto", "cuda"]:
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

    # Datasets
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
    model = NERAttentionModel(
        word_vocab_size=word_vocab_size,
        char_vocab_size=char_vocab_size,
        num_tags=len(TAG_TO_ID),
        word_embedding_dim=100,
        char_embedding_dim=50,
        filter_widths=filter_widths,
        filters_per_width=filters_per_width,
        hidden_dim=128,
        num_layers=bilstm_layers,
        num_heads=num_heads,
        num_attention_blocks=num_attention_blocks,
        attention_dropout=attention_dropout,
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

    global_step = 0
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            global_step += 1

            # Linear LR warmup for attention stability
            if global_step <= warmup_steps and warmup_steps > 0:
                warmup_lr = lr * (global_step / warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr

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
        dev_metrics = evaluate_attention_model(model, dev_loader, device)
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
                    "num_heads": num_heads,
                    "num_attention_blocks": num_attention_blocks,
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
        "num_heads": num_heads,
        "num_attention_blocks": num_attention_blocks,
        "filter_widths": list(filter_widths),
    }
