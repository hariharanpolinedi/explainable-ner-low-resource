"""
Iterative Self-Training / Pseudo-Labeling Loop with Confidence Tightening (Phase 10).

Executes multi-round self-training for low-resource NER:
1. Runs confidence-based inference with CRF normalized confidence on the unlabeled pool.
2. Filters sentences at or above round threshold (starting 0.90, +0.02/round, max 4 rounds).
3. Enforces 3x pseudo-volume cap relative to combined gold+silver volume.
4. Retrains from current best checkpoint with lower learning rate (1e-4) on gold+silver+cumulative_pseudo.
5. Evaluates on gold-dev; automatically rolls back degraded checkpoints.
6. Evaluates stopping rule: 2-round consecutive improvement < 0.5 F1 points or max rounds.
7. Strictly guarantees zero test-set leakage.
"""

from __future__ import annotations

import json
import os
import random
import shutil
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
from evaluation.test_set_isolation_guard import load_sentences_from_conll, verify_pool_isolation
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.confidence_filter import filter_unlabeled_pool, load_model_for_inference
from training.multilingual_dataset import (
    LANG_TO_ID,
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from utils.logger import ExperimentLogger
from utils.seed import set_global_seed


def collect_candidate_pool(
    lang: str,
    processed_dir: Path,
    used_sentences: set[str],
    gazetteer_dir: Path,
    target_pool_size: int = 15000,
) -> list[list[str]]:
    """
    Collect candidate unlabeled sentences from processed corpus,
    excluding all used sentences (gold, silver, prior pseudo rounds).
    Prioritizes entity-bearing candidate sentences alongside general sentences.
    """
    raw_file = processed_dir / lang / "sentences.txt"
    if not raw_file.is_file():
        raise FileNotFoundError(f"Missing processed sentences file: {raw_file}")

    # Load gazetteer terms for prioritization
    gaz_lang_dir = gazetteer_dir / lang
    entity_terms: set[str] = set()
    for fname in ["persons.txt", "places.txt", "orgs.txt"]:
        fpath = gaz_lang_dir / fname
        if fpath.is_file():
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    t = line.strip()
                    if t:
                        entity_terms.add(t)

    entity_candidates: list[list[str]] = []
    general_candidates: list[list[str]] = []

    with open(raw_file, "r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            line_str = line.strip()
            if not line_str:
                continue
            toks = line_str.split()
            if not (3 <= len(toks) <= 60):
                continue
            s_text = " ".join(toks)
            if s_text in used_sentences:
                continue

            has_term = any(term in s_text for term in entity_terms)
            if has_term:
                entity_candidates.append(toks)
            else:
                general_candidates.append(toks)

            if len(entity_candidates) >= target_pool_size * 0.7 and len(general_candidates) >= target_pool_size * 0.3:
                break

    combined = entity_candidates + general_candidates
    random.shuffle(combined)
    return combined[:target_pool_size]


def train_single_round(
    model: NERMultilingualModel,
    train_dataset: MultilingualNERDataset,
    dev_dataset: MultilingualNERDataset,
    device: torch.device,
    lr: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 12,
    patience: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Fine-tune model for one self-training round on combined gold+silver+pseudo dataset.
    Returns (best_model_state_dict, best_eval_metrics).
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_multilingual,
    )

    best_dev_f1 = -1.0
    best_state: dict[str, Any] = {}
    best_metrics: dict[str, Any] = {}
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
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

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Evaluate on dev
        eval_metrics = evaluate_joint_model_on_language(
            model=model,
            dev_dataset=dev_dataset,
            device=device,
            batch_size=batch_size,
        )
        dev_f1 = eval_metrics["entity_f1"]

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_metrics = eval_metrics
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            improved_mark = "(*)"
        else:
            epochs_no_improve += 1
            improved_mark = ""

        print(
            f"    Epoch {epoch:02d}/{max_epochs} | Loss: {avg_loss:.4f} | "
            f"Dev F1: {dev_f1*100:.2f}% (Best: {best_dev_f1*100:.2f}%) {improved_mark}"
        )

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    return best_state, best_metrics


def run_self_training_for_language(
    lang: str,
    gold_dir: Path | str,
    silver_dir: Path | str,
    proc_dir: Path | str,
    gaz_dir: Path | str,
    checkpoints_dir: Path | str,
    pseudo_dir: Path | str,
    logger: ExperimentLogger,
    initial_threshold: float = 0.90,
    threshold_step: float = 0.02,
    max_rounds: int = 4,
    volume_cap_multiplier: float = 3.0,
    per_round_pseudo_target: int = 2000,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    max_epochs_per_round: int = 12,
    patience: int = 3,
    seed: int = 42,
    device_setting: str = "auto",
) -> dict[str, Any]:
    """
    Run complete multi-round self-training pipeline for a single language.
    """
    set_global_seed(seed)
    gold_dir = Path(gold_dir)
    silver_dir = Path(silver_dir)
    proc_dir = Path(proc_dir)
    gaz_dir = Path(gaz_dir)
    checkpoints_dir = Path(checkpoints_dir)
    pseudo_dir = Path(pseudo_dir) / lang
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    if device_setting == "cuda" or (device_setting == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\n========================================================")
    print(f"=== [Phase 10] SELF-TRAINING LOOP: {lang.upper()} (Device: {device}) ===")
    print(f"========================================================")

    # 1. Load Vocabularies
    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)

    # 2. Collect used sentences (Gold + Silver) for isolation guard
    gold_train_path = gold_dir / lang / "train.conll"
    gold_dev_path = gold_dir / lang / "dev.conll"
    gold_test_path = gold_dir / lang / "test.conll"
    silver_train_path = silver_dir / lang / "train.conll"

    gold_train_sents = set(load_sentences_from_conll(gold_train_path))
    gold_dev_sents = set(load_sentences_from_conll(gold_dev_path))
    gold_test_sents = set(load_sentences_from_conll(gold_test_path))
    silver_train_sents = set(load_sentences_from_conll(silver_train_path))

    used_sents = gold_train_sents | gold_dev_sents | gold_test_sents | silver_train_sents

    # 3. Base Volume Cap Calculation
    base_volume = len(gold_train_sents) + len(silver_train_sents)
    max_allowed_pseudo = int(volume_cap_multiplier * base_volume)
    print(f"[{lang.upper()}] Base volume (Gold+Silver): {base_volume:,} | Max pseudo cap (3x): {max_allowed_pseudo:,}")

    # 4. Initialize model from Phase 9 best checkpoint
    p9_checkpoint_path = checkpoints_dir / f"finetuned_{lang}.pt"
    if not p9_checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing Phase 9 checkpoint for {lang}: {p9_checkpoint_path}")

    current_best_model = NERMultilingualModel(
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

    chk = torch.load(p9_checkpoint_path, map_location=device, weights_only=False)
    current_best_model.load_state_dict(chk["model_state_dict"])
    current_best_checkpoint_path = p9_checkpoint_path

    # 5. Dev dataset for evaluation
    dev_dataset = MultilingualNERDataset(
        lang_conll_paths={lang: [gold_dev_path]},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )

    # 6. Baseline Evaluation (Round 0)
    baseline_metrics = evaluate_joint_model_on_language(
        model=current_best_model,
        dev_dataset=dev_dataset,
        device=device,
        batch_size=batch_size,
    )
    baseline_f1 = baseline_metrics["entity_f1"]
    current_best_f1 = baseline_f1
    print(f"[{lang.upper()}] Phase 9 Baseline Dev F1 (Round 0): {baseline_f1*100:.2f}%")

    history: list[dict[str, Any]] = [{
        "round": 0,
        "threshold": None,
        "pseudo_added": 0,
        "cumulative_pseudo": 0,
        "dev_f1": baseline_f1,
        "precision": baseline_metrics["entity_precision"],
        "recall": baseline_metrics["entity_recall"],
        "token_acc": baseline_metrics["token_accuracy"],
        "status": "PHASE_9_BASELINE",
        "action": "ESTABLISHED_BASELINE",
    }]

    cumulative_pseudo_paths: list[Path] = []
    cumulative_pseudo_sents = 0
    all_pseudo_sents_set: set[str] = set()

    stopping_reason = "MAX_ROUNDS_REACHED"
    best_round_num = 0

    # Multi-Round Self-Training Loop
    for r in range(1, max_rounds + 1):
        round_thresholds = {
            "PER": round(min(0.96, initial_threshold + (r - 1) * threshold_step), 4),
            "LOC": round(min(0.96, initial_threshold + (r - 1) * threshold_step), 4),
            "ORG": round(min(0.90, (initial_threshold - 0.15) + (r - 1) * threshold_step), 4),  # 0.75 base
            "MISC": round(min(0.94, (initial_threshold - 0.05) + (r - 1) * threshold_step), 4),
            "O": round(min(0.96, initial_threshold + (r - 1) * threshold_step), 4),
        }
        print(f"\n--- [Round {r}/{max_rounds}] Per-Type Thresholds: {round_thresholds} ---")

        # Step A: Assemble Candidate Pool (Disjoint from Gold, Silver, and Prior Pseudo)
        current_excluded = used_sents | all_pseudo_sents_set
        candidate_tokens = collect_candidate_pool(
            lang=lang,
            processed_dir=proc_dir,
            used_sentences=current_excluded,
            gazetteer_dir=gaz_dir,
            target_pool_size=12000,
        )

        # Pre-inference Test Isolation Check
        cand_sentences_str = [" ".join(toks) for toks in candidate_tokens]
        verify_pool_isolation(lang, cand_sentences_str, gold_dir=gold_dir)

        # Step B: Confidence Filtering Inference
        remaining_budget = max_allowed_pseudo - cumulative_pseudo_sents
        round_cap = min(per_round_pseudo_target, remaining_budget)

        if round_cap <= 0:
            print(f"[{lang.upper()}] Pseudo-data volume cap reached ({cumulative_pseudo_sents:,}/{max_allowed_pseudo:,}).")
            stopping_reason = "VOLUME_CAP_REACHED"
            break

        round_conll_path = pseudo_dir / f"round_{r}.conll"
        filter_res = filter_unlabeled_pool(
            lang=lang,
            candidate_tokens=candidate_tokens,
            checkpoint_path=current_best_checkpoint_path,
            proc_dir=proc_dir,
            threshold=round_thresholds,
            device=device,
            batch_size=128,
            volume_cap=round_cap,
            output_conll_path=round_conll_path,
        )

        num_added = filter_res["retained_count"]
        ent_counts = filter_res.get("per_type_entity_counts", {})
        print(
            f"[{lang.upper()}] Filtered: {filter_res['total_above_threshold']:,} above per-type thresholds. "
            f"Retained (Capped): {num_added:,} sentences (Avg Conf: {filter_res['average_confidence']:.4f}, "
            f"Entity ratio: {filter_res['entity_bearing_ratio']*100:.1f}%, Entities: {ent_counts})"
        )

        if num_added == 0:
            print(f"[{lang.upper()}] Zero sentences met threshold {round_thresholds}. Ending loop.")
            stopping_reason = "THRESHOLD_EXHAUSTED"
            break

        # Register added sentences to exclusion set
        for item in filter_res["retained_samples"]:
            all_pseudo_sents_set.add(" ".join(item["tokens"]))

        cumulative_pseudo_sents += num_added
        cumulative_pseudo_paths.append(round_conll_path)

        # Verify volume cap constraint
        assert cumulative_pseudo_sents <= max_allowed_pseudo, (
            f"Volume cap violated: {cumulative_pseudo_sents} > {max_allowed_pseudo}"
        )

        # Step C: Retrain on Gold + Silver + Cumulative Pseudo
        train_conll_inputs = [gold_train_path, silver_train_path] + cumulative_pseudo_paths
        train_dataset = MultilingualNERDataset(
            lang_conll_paths={lang: train_conll_inputs},
            word_to_id=word_to_id,
            char_to_id=char_to_id,
        )

        print(
            f"[{lang.upper()}] Retraining with LR={learning_rate} on {len(train_dataset):,} sentences "
            f"(Gold: 600, Silver: 7,200, Pseudo: {cumulative_pseudo_sents:,})..."
        )

        # Reload best model weights as starting point for this round
        round_model = NERMultilingualModel(
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
        best_chk = torch.load(current_best_checkpoint_path, map_location=device, weights_only=False)
        round_model.load_state_dict(best_chk["model_state_dict"])

        round_state, round_metrics = train_single_round(
            model=round_model,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            device=device,
            lr=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs_per_round,
            patience=patience,
        )

        round_dev_f1 = round_metrics["entity_f1"]
        delta_vs_prev = round_dev_f1 - history[-1]["dev_f1"]
        delta_vs_base = round_dev_f1 - baseline_f1

        print(
            f"[{lang.upper()}] Round {r} Result: Dev F1 = {round_dev_f1*100:.2f}% "
            f"(Delta vs prev: {delta_vs_prev*100:+.2f}%, Delta vs P9: {delta_vs_base*100:+.2f}%)"
        )

        # Step D: Regression Check & Rollback Logic
        round_checkpoint_path = checkpoints_dir / f"selftrained_{lang}_round{r}.pt"

        if round_dev_f1 >= current_best_f1:
            status = "ACCEPTED (IMPROVED/MAINTAINED)"
            action = "UPDATED_BEST"
            current_best_f1 = round_dev_f1
            best_round_num = r

            # Save round checkpoint
            torch.save({
                "round": r,
                "language": lang,
                "model_state_dict": round_state,
                "metrics": round_metrics,
                "threshold": round_threshold,
                "pseudo_volume": cumulative_pseudo_sents,
                "seed": seed,
            }, round_checkpoint_path)

            current_best_checkpoint_path = round_checkpoint_path
            # Save as current best
            best_target_path = checkpoints_dir / f"selftrained_{lang}_best.pt"
            shutil.copyfile(round_checkpoint_path, best_target_path)
            print(f"[{lang.upper()}] Promoted Round {r} checkpoint -> {best_target_path.name}")
        else:
            status = "REGRESSION_DETECTED (ROLLED BACK)"
            action = "ROLLED_BACK_TO_PREVIOUS"
            print(
                f"[{lang.upper()}] Dev F1 dropped from {current_best_f1*100:.2f}% to {round_dev_f1*100:.2f}%. "
                f"ROLLING BACK to previous best checkpoint ({current_best_checkpoint_path.name})."
            )

        history_entry = {
            "round": r,
            "threshold": round_threshold,
            "pseudo_added": num_added,
            "cumulative_pseudo": cumulative_pseudo_sents,
            "dev_f1": round_dev_f1,
            "precision": round_metrics.get("entity_precision", 0.0),
            "recall": round_metrics.get("entity_recall", 0.0),
            "token_acc": round_metrics.get("token_accuracy", 0.0),
            "status": status,
            "action": action,
        }
        history.append(history_entry)

        # Step E: Stopping Rule Evaluation
        # Stop when dev-F1 improvement over two consecutive rounds is < 0.5 points combined
        if r >= 2:
            two_round_gain = (history[r]["dev_f1"] - history[r - 2]["dev_f1"]) * 100
            print(f"[{lang.upper()}] Two-round cumulative improvement: {two_round_gain:+.2f} F1 points")
            if two_round_gain < 0.5:
                stopping_reason = f"PLATEAU_REACHED (2-round improvement {two_round_gain:.2f} < 0.5 pts)"
                print(f"[{lang.upper()}] Stopping Rule Triggered: {stopping_reason}")
                break

    # If best round was Round 0, ensure selftrained_{lang}_best.pt is linked/copied from p9
    final_best_path = checkpoints_dir / f"selftrained_{lang}_best.pt"
    if not final_best_path.is_file():
        shutil.copyfile(p9_checkpoint_path, final_best_path)

    # Log results
    logger.log_run(
        config_path="configs/base_config.yaml",
        metrics={
            "phase": 10,
            "language": lang,
            "baseline_p9_dev_f1": baseline_f1,
            "best_dev_f1": current_best_f1,
            "best_round": best_round_num,
            "delta_dev_f1": current_best_f1 - baseline_f1,
            "total_pseudo_volume": cumulative_pseudo_sents,
            "volume_cap": max_allowed_pseudo,
            "stopping_reason": stopping_reason,
        },
        dataset_version=f"phase10-selftraining-{lang}",
        extra_metadata={"history": history},
    )

    print(
        f"\n[{lang.upper()}] Finished Self-Training: Baseline {baseline_f1*100:.2f}% -> "
        f"Best {current_best_f1*100:.2f}% at Round {best_round_num} "
        f"(Total Pseudo Added: {cumulative_pseudo_sents:,}) | Reason: {stopping_reason}"
    )

    return {
        "language": lang,
        "baseline_f1": baseline_f1,
        "best_dev_f1": current_best_f1,
        "best_round": best_round_num,
        "delta_f1": current_best_f1 - baseline_f1,
        "total_pseudo_volume": cumulative_pseudo_sents,
        "volume_cap": max_allowed_pseudo,
        "stopping_reason": stopping_reason,
        "history": history,
    }


def run_all_languages_self_training(
    config_path: str = "configs/base_config.yaml",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Execute complete Phase 10 self-training across all configured languages."""
    from utils.config_loader import load_config
    cfg = load_config(config_path)

    logger = ExperimentLogger(log_dir=cfg.logging.log_dir, backend=cfg.logging.backend)
    gold_dir = PROJECT_ROOT / "data" / "gold"
    silver_dir = PROJECT_ROOT / "data" / "silver"
    proc_dir = PROJECT_ROOT / "data" / "processed"
    gaz_dir = PROJECT_ROOT / "data" / "gazetteers"
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    pseudo_dir = PROJECT_ROOT / "data" / "pseudo"

    init_thresh = cfg.self_training.initial_threshold if cfg.self_training else 0.90
    thresh_step = cfg.self_training.threshold_step if cfg.self_training else 0.02
    max_rounds = cfg.self_training.max_rounds if cfg.self_training else 4
    vol_cap_mult = cfg.self_training.volume_cap_multiplier if cfg.self_training else 3.0
    st_lr = cfg.self_training.learning_rate if cfg.self_training else 1e-4

    results_by_lang: dict[str, list[dict[str, Any]]] = {}
    summary_by_lang: dict[str, dict[str, Any]] = {}

    for lang in cfg.project.languages:
        res = run_self_training_for_language(
            lang=lang,
            gold_dir=gold_dir,
            silver_dir=silver_dir,
            proc_dir=proc_dir,
            gaz_dir=gaz_dir,
            checkpoints_dir=checkpoints_dir,
            pseudo_dir=pseudo_dir,
            logger=logger,
            initial_threshold=init_thresh,
            threshold_step=thresh_step,
            max_rounds=max_rounds,
            volume_cap_multiplier=vol_cap_mult,
            learning_rate=st_lr,
            seed=cfg.seed,
        )
        results_by_lang[lang] = res["history"]
        summary_by_lang[lang] = res

    # Generate figures and reports
    from evaluation.plot_selftraining_curve import generate_selftraining_report, plot_selftraining_curve
    plot_selftraining_curve(results_by_lang)
    generate_selftraining_report(results_by_lang, summary_by_lang)

    return results_by_lang, summary_by_lang


if __name__ == "__main__":
    run_all_languages_self_training()
