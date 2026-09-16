"""
Evaluation and Comparison Pipeline: Multilingual Joint Model vs Phase-7 Monolingual Baselines (Phase 8).

Evaluates a joint model checkpoint on each language's own gold-dev set separately,
computing strict span-level entity F1, precision, recall, and token accuracy via seqeval,
and directly comparing each language's performance against its Phase-7 monolingual baseline.
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
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore

from evaluation.seqeval_metrics import compute_seqeval_metrics
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import (
    ID_TO_LANG,
    LANG_TO_ID,
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from training.train_charcnn import ID_TO_TAG


def evaluate_joint_model_on_language(
    model: nn.Module,
    dev_dataset: MultilingualNERDataset,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    Evaluate a multilingual model on a single language's dev dataset.
    """
    model.eval()
    dataloader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_multilingual,
    )

    all_gold_tags: list[list[str]] = []
    all_pred_tags: list[list[str]] = []

    with torch.no_grad():
        for batch in dataloader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)
            seq_lengths = batch["seq_lengths"]

            paths = model(word_ids, char_ids, lang_ids=lang_ids, mask=mask)

            for i in range(len(seq_lengths)):
                slen = seq_lengths[i]
                gold_seq = [ID_TO_TAG[tid.item()] for tid in tags[i, :slen]]
                pred_seq = [ID_TO_TAG[tid] for tid in paths[i][:slen]]
                all_gold_tags.append(gold_seq)
                all_pred_tags.append(pred_seq)

    metrics = compute_seqeval_metrics(all_gold_tags, all_pred_tags)
    return {
        "entity_f1": metrics["overall"]["entity_f1"],
        "entity_precision": metrics["overall"]["entity_precision"],
        "entity_recall": metrics["overall"]["entity_recall"],
        "token_accuracy": metrics["overall"]["token_accuracy"],
        "overall": metrics["overall"],
        "by_type": metrics["by_type"],
    }


def evaluate_all_languages_joint(
    model: nn.Module,
    gold_dir: Path,
    proc_dir: Path,
    word_to_id: dict[str, int],
    char_to_id: dict[str, int],
    device: torch.device,
    languages: tuple[str, ...] = ("konkani", "assamese", "sinhala"),
    batch_size: int = 32,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate joint model across all languages separately on each language's gold-dev set.
    """
    results: dict[str, dict[str, Any]] = {}
    for lang in languages:
        dev_file = gold_dir / lang / "dev.conll"
        dev_ds = MultilingualNERDataset(
            lang_conll_paths={lang: [dev_file]},
            word_to_id=word_to_id,
            char_to_id=char_to_id,
        )
        res = evaluate_joint_model_on_language(model, dev_ds, device, batch_size=batch_size)
        results[lang] = res
    return results


def load_phase7_baselines(checkpoints_dir: Path) -> dict[str, float]:
    """
    Load Phase 7 monolingual baseline entity-F1 scores from checkpoints.
    """
    baselines: dict[str, float] = {}
    for lang in ["konkani", "assamese", "sinhala"]:
        chk_path = checkpoints_dir / f"attention_{lang}.pt"
        if chk_path.is_file():
            chk = torch.load(chk_path, map_location="cpu", weights_only=False)
            baselines[lang] = chk.get("best_dev_f1", 0.0)
        else:
            baselines[lang] = 0.0
    return baselines
