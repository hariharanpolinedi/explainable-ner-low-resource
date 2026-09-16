"""
Per-Language Fine-Tuning and Monolingual Control Pipeline (Phase 9).

Specializes the Phase-8 joint multilingual checkpoint per language using
a progressive layer-unfreezing schedule, and trains from-scratch monolingual
controls for fair empirical comparison.
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

from evaluation.compare_joint_vs_monolingual import evaluate_joint_model_on_language
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.layer_unfreeze_schedule import LayerUnfreezeScheduler
from training.multilingual_dataset import (
    LANG_TO_ID,
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from utils.seed import set_global_seed


def train_finetune_experiment(
    lang: str,
    gold_dir: Path | str,
    silver_dir: Path | str,
    proc_dir: Path | str,
    checkpoint_out: Path | str,
    init_checkpoint_path: Path | str | None = None,
    schedule: str = "phased",
    unfreeze_epoch: int = 4,
    lr: float = 2e-4,
    warmup_steps: int = 50,
    batch_size: int = 32,
    max_epochs: int = 20,
    patience: int = 5,
    seed: int = 42,
    device_setting: str = "auto",
) -> dict[str, Any]:
    """
    Train either a fine-tuned model (from Phase 8 joint_best.pt) or a from-scratch control model.
    """
    if torch is None:
        raise ImportError("PyTorch is required for train_finetune_experiment.")

    set_global_seed(seed)

    if device_setting == "cuda" or (device_setting == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    gold_dir = Path(gold_dir)
    silver_dir = Path(silver_dir)
    proc_dir = Path(proc_dir)
    checkpoint_out = Path(checkpoint_out)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    is_control = (init_checkpoint_path is None)
    exp_mode = "From-Scratch Control" if is_control else f"Fine-Tuned ({schedule})"

    # 1. Load unified multilingual vocabularies
    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)

    # 2. Build single-language training dataset (gold + silver)
    gold_train = gold_dir / lang / "train.conll"
    silver_train = silver_dir / lang / "train.conll"
    dev_path = gold_dir / lang / "dev.conll"

    train_dataset = MultilingualNERDataset(
        lang_conll_paths={lang: [gold_train, silver_train]},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )

    dev_dataset = MultilingualNERDataset(
        lang_conll_paths={lang: [dev_path]},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_multilingual,
    )

    # 3. Instantiate model
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

    # 4. Checkpoint Loading (if fine-tuning)
    if not is_control:
        init_path = Path(init_checkpoint_path)
        if not init_path.is_file():
            raise FileNotFoundError(f"Missing joint checkpoint to fine-tune from: {init_path}")
        chk = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(chk["model_state_dict"])
        print(f"Loaded Phase 8 initialization weights from {init_path.name}")
    else:
        print("Initialized model with fresh random weights (Control run).")

    # 5. Layer Unfreeze Scheduler
    unfreeze_scheduler = LayerUnfreezeScheduler(
        model=model,
        schedule=schedule,
        unfreeze_epoch=unfreeze_epoch,
    )

    # Optimizer: AdamW over all model parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Linear learning rate warmup
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    print(f"\n[Phase 9] Language: {lang.upper()} | Mode: {exp_mode}")
    print(f"Dataset: {len(train_dataset)} sentences, Device: {device}")
    print(f"Schedule: {schedule} (unfreeze at epoch {unfreeze_epoch}), LR: {lr}, Max Epochs: {max_epochs}")

    best_dev_f1 = -1.0
    best_epoch = -1
    best_metrics: dict[str, Any] = {}
    epochs_no_improve = 0
    training_start = time.time()
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Step unfreezing schedule
        freeze_status = unfreeze_scheduler.step_epoch(epoch)

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
            lr_scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start

        # Evaluate on dev
        eval_metrics = evaluate_joint_model_on_language(
            model=model,
            dev_dataset=dev_dataset,
            device=device,
            batch_size=batch_size,
        )

        dev_f1 = eval_metrics["entity_f1"]
        dev_acc = eval_metrics["token_accuracy"]

        history_entry = {
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "dev_f1": dev_f1,
            "token_accuracy": dev_acc,
            "lower_frozen": freeze_status["lower_layers_frozen"],
            "epoch_duration_s": round(epoch_time, 1),
        }
        history.append(history_entry)

        status_str = "FROZEN" if freeze_status["lower_layers_frozen"] else "UNFROZEN"
        print(
            f"Epoch {epoch:02d}/{max_epochs} [{status_str}] | "
            f"Loss: {avg_loss:.4f} | Dev F1: {dev_f1*100:.2f}% | "
            f"Token Acc: {dev_acc*100:.2f}% | Time: {epoch_time:.1f}s"
        )

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch = epoch
            best_metrics = eval_metrics
            epochs_no_improve = 0

            # Save checkpoint
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dev_f1": best_dev_f1,
                "best_metrics": best_metrics,
                "language": lang,
                "schedule": schedule,
                "unfreeze_epoch": unfreeze_epoch,
                "lr": lr,
                "is_control": is_control,
                "seed": seed,
            }
            torch.save(payload, checkpoint_out)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    total_time = time.time() - training_start
    print(
        f"Finished: Best Dev F1 = {best_dev_f1*100:.2f}% at Epoch {best_epoch} "
        f"in {total_time:.1f}s -> Saved to {checkpoint_out.name}"
    )

    return {
        "language": lang,
        "is_control": is_control,
        "schedule": schedule,
        "best_epoch": best_epoch,
        "best_dev_f1": best_dev_f1,
        "best_metrics": best_metrics,
        "history": history,
        "total_time_seconds": total_time,
        "checkpoint_path": str(checkpoint_out),
    }
