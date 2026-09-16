"""
Training and evaluation pipeline for Word-Level BiLSTM-CRF baseline NER models.

Supports:
- Gold-only vs Gold+Silver training conditions (Experiment A)
- 1-layer vs 2-layer BiLSTM architectures (Experiment B)
- Early stopping on gold-dev entity-level F1
- Checkpoint persistence to checkpoints/baseline_{lang}.pt
- Experiment logging with device tracking
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

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
from model.ner_model_v1_baseline import NERBaselineModel
from utils.config_loader import BaseConfig, load_config, parse_cli_args
from utils.logger import ExperimentLogger
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


class NERCoNLLDataset(Dataset):
    """Dataset for CoNLL formatted NER data with vocabulary index mapping."""

    def __init__(
        self,
        conll_paths: list[Path | str],
        word_to_id: dict[str, int],
        pad_token_id: int = 0,
        unk_token_id: int = 1,
    ) -> None:
        self.word_to_id = word_to_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.samples: list[tuple[list[str], list[int], list[int]]] = []
        self._load_files(conll_paths)

    def _load_files(self, paths: list[Path | str]) -> None:
        for p in paths:
            fpath = Path(p)
            if not fpath.is_file():
                raise FileNotFoundError(f"Missing CoNLL file: {fpath}")

            curr_tokens: list[str] = []
            curr_tag_ids: list[int] = []

            with open(fpath, "r", encoding="utf-8") as fp:
                for line in fp:
                    line_str = line.strip()
                    if not line_str:
                        if curr_tokens:
                            token_ids = [self.word_to_id.get(t, self.unk_token_id) for t in curr_tokens]
                            self.samples.append((curr_tokens, token_ids, curr_tag_ids))
                            curr_tokens = []
                            curr_tag_ids = []
                        continue

                    parts = line_str.split("\t")
                    if len(parts) == 2:
                        tok, tag = parts
                        curr_tokens.append(tok)
                        curr_tag_ids.append(TAG_TO_ID.get(tag, 0))

            if curr_tokens:
                token_ids = [self.word_to_id.get(t, self.unk_token_id) for t in curr_tokens]
                self.samples.append((curr_tokens, token_ids, curr_tag_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[str], list[int], list[int]]:
        return self.samples[idx]


def collate_fn(batch: list[tuple[list[str], list[int], list[int]]]) -> dict[str, Any]:
    """Pads batch sequences dynamically to max sequence length in batch."""
    batch_tokens = [item[0] for item in batch]
    token_ids_list = [item[1] for item in batch]
    tag_ids_list = [item[2] for item in batch]

    lengths = [len(seq) for seq in token_ids_list]
    max_len = max(lengths) if lengths else 1

    padded_tokens = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_tags = torch.zeros((len(batch), max_len), dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, (tok_ids, tag_ids) in enumerate(zip(token_ids_list, tag_ids_list)):
        l = len(tok_ids)
        if l > 0:
            padded_tokens[i, :l] = torch.tensor(tok_ids, dtype=torch.long)
            padded_tags[i, :l] = torch.tensor(tag_ids, dtype=torch.long)
            mask[i, :l] = True

    return {
        "raw_tokens": batch_tokens,
        "input_ids": padded_tokens,
        "tags": padded_tags,
        "mask": mask,
        "lengths": lengths,
    }


def evaluate_model(
    model: NERBaselineModel,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate model on dataset using seqeval entity F1 and token accuracy."""
    model.eval()
    all_gold_tags: list[list[str]] = []
    all_pred_tags: list[list[str]] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            tags = batch["tags"]
            lengths = batch["lengths"]

            decoded_paths = model(input_ids, mask=mask)

            for b in range(len(lengths)):
                seq_len = lengths[b]
                gold_seq = [ID_TO_TAG[tag_id] for tag_id in tags[b, :seq_len].tolist()]
                pred_seq = [ID_TO_TAG[tag_id] for tag_id in decoded_paths[b]]

                all_gold_tags.append(gold_seq)
                all_pred_tags.append(pred_seq)

    metrics = compute_seqeval_metrics(all_gold_tags, all_pred_tags)
    return metrics


def train_single_experiment(
    lang: str,
    train_paths: list[Path | str],
    dev_path: Path | str,
    vocab_path: Path | str,
    checkpoint_out: Path | str,
    bilstm_layers: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 5,
    device_name: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    """Train a BiLSTM-CRF baseline model and return training statistics and best dev metrics."""
    set_global_seed(seed)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    # Load word vocabulary
    with open(vocab_path, "r", encoding="utf-8") as fp:
        vocab_data = json.load(fp)
    word_to_id = vocab_data["token_to_id"]
    vocab_size = len(word_to_id)

    # Prepare datasets & loaders
    train_dataset = NERCoNLLDataset(train_paths, word_to_id)
    dev_dataset = NERCoNLLDataset([dev_path], word_to_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Initialize model from scratch
    model = NERBaselineModel(
        vocab_size=vocab_size,
        num_tags=len(TAG_TO_ID),
        embedding_dim=100,
        hidden_dim=128,
        num_layers=bilstm_layers,
        dropout=0.3,
        padding_idx=0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_dev_f1 = -1.0
    best_metrics: dict[str, Any] = {}
    patience_counter = 0
    epoch_history: list[dict[str, Any]] = []

    start_time = time.time()
    initial_loss: float | None = None
    final_loss: float | None = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, tags=tags, mask=mask)

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss encountered at epoch {epoch}!")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        if epoch == 1:
            initial_loss = avg_loss
        final_loss = avg_loss

        # Evaluate on dev set
        dev_eval = evaluate_model(model, dev_loader, device)
        dev_f1 = dev_eval["overall"]["entity_f1"]
        token_acc = dev_eval["overall"]["token_accuracy"]

        epoch_history.append({
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "dev_entity_f1": dev_f1,
            "dev_token_acc": token_acc,
        })

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_metrics = dev_eval
            patience_counter = 0
            # Save checkpoint
            checkpoint_path = Path(checkpoint_out)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": vocab_size,
                "num_tags": len(TAG_TO_ID),
                "bilstm_layers": bilstm_layers,
                "best_dev_f1": best_dev_f1,
                "metrics": dev_eval,
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    training_duration = time.time() - start_time
    return {
        "language": lang,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "epochs_trained": len(epoch_history),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decreased": (final_loss < initial_loss) if (initial_loss is not None and final_loss is not None) else False,
        "best_dev_entity_f1": best_dev_f1,
        "best_dev_token_acc": best_metrics.get("overall", {}).get("token_accuracy", 0.0),
        "best_metrics": best_metrics,
        "epoch_history": epoch_history,
        "duration_seconds": round(training_duration, 2),
    }
