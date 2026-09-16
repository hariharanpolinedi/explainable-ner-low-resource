"""
Multilingual Joint Warm-Start Training Pipeline (Phase 8).

Trains a shared Char-CNN + BiLSTM + Self-Attention NER backbone across
Konkani, Assamese, and Sinhala with language conditioning from fresh random initialization.

Supports the full 2x2 experimental matrix:
- CRF-head configurations: 'shared' vs 'per_language'
- Batch sampling strategies: 'uniform' vs 'proportional'
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

from evaluation.compare_joint_vs_monolingual import evaluate_all_languages_joint
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import (
    ID_TO_LANG,
    LANG_TO_ID,
    MultilingualBatchSampler,
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from utils.seed import set_global_seed


def train_multilingual_joint_experiment(
    gold_dir: Path | str,
    silver_dir: Path | str,
    proc_dir: Path | str,
    checkpoint_out: Path | str,
    crf_config: str = "shared",
    sampling_strategy: str = "uniform",
    languages: Sequence[str] = ("konkani", "assamese", "sinhala"),
    lang_embedding_dim: int = 16,
    filter_widths: Sequence[int] = (3, 4, 5),
    filters_per_width: int = 30,
    bilstm_layers: int = 2,
    num_heads: int = 4,
    num_attention_blocks: int = 1,
    attention_dropout: float = 0.1,
    warmup_steps: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_epochs: int = 25,
    patience: int = 4,
    seed: int = 42,
    device_setting: str = "auto",
) -> dict[str, Any]:
    """
    Train a multilingual joint model from fresh random initialization.
    """
    if torch is None:
        raise ImportError("PyTorch is required for train_multilingual_joint_experiment.")

    set_global_seed(seed)

    # Resolve hardware device
    if device_setting == "cuda" or (device_setting == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    gold_dir = Path(gold_dir)
    silver_dir = Path(silver_dir)
    proc_dir = Path(proc_dir)
    checkpoint_out = Path(checkpoint_out)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load or construct unified multilingual vocabularies
    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir, languages=languages)

    # 2. Build pooled training dataset: gold-train + silver for all languages
    train_paths: dict[str, list[Path | str]] = {}
    for lang in languages:
        train_paths[lang] = [
            gold_dir / lang / "train.conll",
            silver_dir / lang / "train.conll",
        ]

    train_dataset = MultilingualNERDataset(
        lang_conll_paths=train_paths,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )

    # 3. Setup batch sampler
    sampler = MultilingualBatchSampler(
        lang_indices=train_dataset.lang_indices,
        batch_size=batch_size,
        strategy=sampling_strategy,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn_multilingual,
    )

    # 4. Instantiate model from fresh random initialization
    model = NERMultilingualModel(
        word_vocab_size=len(word_to_id),
        char_vocab_size=len(char_to_id),
        num_tags=9,
        num_languages=len(languages),
        word_embedding_dim=100,
        char_embedding_dim=50,
        lang_embedding_dim=lang_embedding_dim,
        filter_widths=filter_widths,
        filters_per_width=filters_per_width,
        hidden_dim=128,
        num_layers=bilstm_layers,
        num_heads=num_heads,
        num_attention_blocks=num_attention_blocks,
        attention_dropout=attention_dropout,
        crf_config=crf_config,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Linear learning rate warmup
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    print(f"\n[Phase 8 Joint Training] Config: CRF={crf_config}, Sampling={sampling_strategy}")
    print(f"Dataset: {len(train_dataset)} pooled samples across {list(languages)}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Batches per epoch: {len(sampler)}, Batch Size: {batch_size}, Max Epochs: {max_epochs}")

    best_avg_dev_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    global_step = 0
    training_start = time.time()
    best_eval_results: dict[str, dict[str, Any]] = {}

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            loss = model(word_ids, char_ids, lang_ids=lang_ids, tags=tags, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

        avg_loss = total_loss / max(num_batches, 1)
        epoch_duration = time.time() - epoch_start

        # Evaluate on each language's gold-dev separately
        eval_results = evaluate_all_languages_joint(
            model=model,
            gold_dir=gold_dir,
            proc_dir=proc_dir,
            word_to_id=word_to_id,
            char_to_id=char_to_id,
            device=device,
            languages=tuple(languages),
            batch_size=batch_size,
        )

        avg_dev_f1 = sum(res["entity_f1"] for res in eval_results.values()) / len(eval_results)
        per_lang_f1_str = ", ".join(f"{l}: {eval_results[l]['entity_f1']*100:.2f}%" for l in languages)

        print(
            f"Epoch {epoch:02d}/{max_epochs} | Loss: {avg_loss:.4f} | "
            f"Avg Dev F1: {avg_dev_f1*100:.2f}% ({per_lang_f1_str}) | "
            f"Time: {epoch_duration:.1f}s"
        )

        if avg_dev_f1 > best_avg_dev_f1:
            best_avg_dev_f1 = avg_dev_f1
            best_epoch = epoch
            best_eval_results = eval_results
            epochs_no_improve = 0

            # Save checkpoint
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_avg_dev_f1": best_avg_dev_f1,
                "per_lang_results": best_eval_results,
                "crf_config": crf_config,
                "sampling_strategy": sampling_strategy,
                "word_vocab_size": len(word_to_id),
                "char_vocab_size": len(char_to_id),
                "lang_embedding_dim": lang_embedding_dim,
                "num_heads": num_heads,
                "num_attention_blocks": num_attention_blocks,
                "filter_widths": filter_widths,
                "filters_per_width": filters_per_width,
                "seed": seed,
            }
            torch.save(checkpoint_payload, checkpoint_out)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    total_time = time.time() - training_start
    print(
        f"Completed: Best Avg Dev F1 = {best_avg_dev_f1*100:.2f}% at Epoch {best_epoch} "
        f"in {total_time:.1f}s. Saved to {checkpoint_out}"
    )

    return {
        "crf_config": crf_config,
        "sampling_strategy": sampling_strategy,
        "best_epoch": best_epoch,
        "best_avg_dev_f1": best_avg_dev_f1,
        "per_lang_results": best_eval_results,
        "checkpoint_path": str(checkpoint_out),
        "total_time_seconds": total_time,
        "device": str(device),
    }
