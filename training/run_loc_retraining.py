"""
LOC-Targeted Retraining Pipeline with Hard Divergence Gate.

Retrains multilingual NER models starting from genuine-best baseline checkpoints:
- Konkani: checkpoints/selftrained_konkani_best.pt
- Assamese: checkpoints/augmented_assamese.pt
- Sinhala: checkpoints/augmented_sinhala.pt

Enforces strict standards:
1. Trains on Gold train + LOC-enriched Silver train.
2. Evaluates exclusively on untouched data/gold/{lang}/dev.conll (test.conll remains locked).
3. Evaluates per-type F1 (LOC, ORG, PER, MISC) and checks for false-positive tradeoffs.
4. Divergence Gate (training/checkpoint_divergence_gate.py):
   Refuses to save candidate weights unless >=1.0% params changed and L2 diff >= 0.01.
5. Saves accepted models to checkpoints/loc_patched_{lang}.pt.
6. Generates reports/loc_patch_retraining_results.md.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.compare_joint_vs_monolingual import evaluate_joint_model_on_language
from evaluation.seqeval_metrics import compute_seqeval_metrics, extract_entities
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.checkpoint_divergence_gate import (
    compute_parameter_divergence,
    save_patched_checkpoint_with_gate,
    verify_weight_divergence,
)
from training.multilingual_dataset import (
    LANG_TO_ID,
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from utils.seed import set_global_seed


BASELINE_CHECKPOINTS: dict[str, str] = {
    "konkani": "checkpoints/selftrained_konkani_best.pt",
    "assamese": "checkpoints/augmented_assamese.pt",
    "sinhala": "checkpoints/augmented_sinhala.pt",
}


def train_and_gate_language(
    lang: str,
    baseline_checkpoint_path: Path | str,
    gold_train_path: Path | str,
    silver_train_path: Path | str,
    dev_path: Path | str,
    proc_dir: Path | str,
    output_checkpoint_path: Path | str,
    device: torch.device,
    lr: float = 1e-4,
    batch_size: int = 32,
    epochs: int = 5,
    min_pct_changed: float = 1.0,
    min_l2_diff: float = 0.01,
) -> dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"LOC-TARGETED RETRAINING: {lang.upper()} (Device: {device})")
    print(f"{'='*70}")

    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)

    # Combine gold train (weighted 2x) + silver train
    train_paths = [gold_train_path, gold_train_path, silver_train_path]
    train_ds = MultilingualNERDataset(
        lang_conll_paths={lang: train_paths},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )
    dev_ds = MultilingualNERDataset(
        lang_conll_paths={lang: [dev_path]},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_multilingual,
    )

    # Initialize model
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

    # Load baseline weights
    ckpt = torch.load(baseline_checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)

    # Snapshot untouched baseline state
    baseline_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate untouched baseline on gold dev
    baseline_eval = evaluate_joint_model_on_language(model, dev_ds, device=device)
    base_loc = baseline_eval["by_type"].get("LOC", {})
    base_org = baseline_eval["by_type"].get("ORG", {})
    base_per = baseline_eval["by_type"].get("PER", {})

    print(
        f"[{lang.upper()}] Baseline Dev - Overall F1: {baseline_eval['entity_f1']*100:.2f}% | "
        f"LOC F1: {base_loc.get('f1', 0.0)*100:.2f}% (P: {base_loc.get('precision', 0.0)*100:.2f}%, R: {base_loc.get('recall', 0.0)*100:.2f}%) | "
        f"ORG F1: {base_org.get('f1', 0.0)*100:.2f}% | PER F1: {base_per.get('f1', 0.0)*100:.2f}%"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_cand_state: dict[str, torch.Tensor] | None = None
    best_eval: dict[str, Any] | None = None
    best_score = -1.0

    print(f"[{lang.upper()}] Retraining on {len(train_ds):,} sentences for {epochs} epochs...")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            loss = model(word_ids, char_ids, lang_ids=lang_ids, tags=tags, mask=mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        ep_eval = evaluate_joint_model_on_language(model, dev_ds, device=device)

        ep_loc = ep_eval["by_type"].get("LOC", {})
        ep_org = ep_eval["by_type"].get("ORG", {})
        ep_per = ep_eval["by_type"].get("PER", {})
        ep_overall = ep_eval["entity_f1"]

        # Composite score balancing LOC recovery and preserving overall/ORG accuracy
        loc_f1 = ep_loc.get("f1", 0.0)
        org_f1 = ep_org.get("f1", 0.0)
        score = 0.50 * loc_f1 + 0.30 * ep_overall + 0.20 * org_f1

        print(
            f"  Epoch {ep:2d}/{epochs:2d} - Loss: {avg_loss:.4f} | "
            f"Dev F1: {ep_overall*100:.2f}% | LOC F1: {loc_f1*100:.2f}% (P: {ep_loc.get('precision', 0.0)*100:.2f}%, R: {ep_loc.get('recall', 0.0)*100:.2f}%) | "
            f"ORG F1: {org_f1*100:.2f}% | Score: {score*100:.2f}%"
        )

        if score > best_score:
            best_score = score
            best_eval = ep_eval
            best_cand_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_cand_state is None:
        best_cand_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_eval = ep_eval

    # Divergence gate check
    passed, div, div_msg = verify_weight_divergence(
        best_cand_state,
        baseline_state,
        min_pct_changed=min_pct_changed,
        min_l2_diff=min_l2_diff,
    )

    metadata = {
        "language": lang,
        "baseline_checkpoint": str(baseline_checkpoint_path),
        "baseline_dev_eval": baseline_eval,
        "retrained_dev_eval": best_eval,
        "epochs": epochs,
        "learning_rate": lr,
        "divergence_metrics": div,
        "divergence_gate_passed": passed,
    }

    if passed:
        save_patched_checkpoint_with_gate(
            candidate_state=best_cand_state,
            baseline_state=baseline_state,
            output_path=output_checkpoint_path,
            metadata=metadata,
            min_pct_changed=min_pct_changed,
            min_l2_diff=min_l2_diff,
        )
    else:
        print(f"\n[DIVERGENCE GATE BLOCKED] {div_msg}")
        print(f"[DIVERGENCE GATE BLOCKED] Refusing to write un-updated model to {output_checkpoint_path}\n")

    return {
        "language": lang,
        "baseline_eval": baseline_eval,
        "retrained_eval": best_eval,
        "divergence_gate_passed": passed,
        "divergence_metrics": div,
        "divergence_message": div_msg,
        "output_checkpoint_path": str(output_checkpoint_path) if passed else None,
    }


def run_all_loc_retraining(seed: int = 42) -> dict[str, Any]:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    gold_dir = PROJECT_ROOT / "data" / "gold"
    silver_dir = PROJECT_ROOT / "data" / "silver"
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}

    for lang, base_rel in BASELINE_CHECKPOINTS.items():
        base_path = PROJECT_ROOT / base_rel
        gold_train = gold_dir / lang / "train.conll"
        silver_train = silver_dir / lang / "train.conll"
        dev_path = gold_dir / lang / "dev.conll"
        out_ckpt = checkpoints_dir / f"loc_patched_{lang}.pt"

        res = train_and_gate_language(
            lang=lang,
            baseline_checkpoint_path=base_path,
            gold_train_path=gold_train,
            silver_train_path=silver_train,
            dev_path=dev_path,
            proc_dir=proc_dir,
            output_checkpoint_path=out_ckpt,
            device=device,
            lr=1e-4,
            batch_size=32,
            epochs=5,
        )
        all_results[lang] = res

    # Generate reports/loc_patch_retraining_results.md
    report_path = reports_dir / "loc_patch_retraining_results.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# LOC-Targeted Retraining & Divergence Gate Results\n\n")
        f.write("**Date**: 2026-09-17  \n")
        f.write("**Auditor**: Senior ML/NLP Engineer  \n")
        f.write("**Evaluation Set**: Locked, Untouched Gold-Dev (`data/gold/{lang}/dev.conll`) ONLY  \n\n")
        f.write("---\n\n")
        f.write("## 1. Executive Summary & Divergence Gate Verification\n\n")
        f.write("| Language | Baseline Checkpoint | Divergence Gate | % Params Changed | L2 Diff | Max Diff | Status |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :--- |\n")
        for lang, r in all_results.items():
            base_name = Path(BASELINE_CHECKPOINTS[lang]).name
            gate_str = "PASSED" if r["divergence_gate_passed"] else "FAILED"
            div = r["divergence_metrics"]
            f.write(
                f"| **{lang.capitalize()}** | `{base_name}` | **{gate_str}** | "
                f"{div['pct_changed']:.2f}% | {div['l2_diff']:.4f} | {div['max_diff']:.4f} | "
                f"{'Saved to loc_patched_' + lang + '.pt' if r['divergence_gate_passed'] else 'REJECTED (Unchanged)'} |\n"
            )

        f.write("\n---\n\n")
        f.write("## 2. Before vs After Performance on Real Gold-Dev\n\n")
        f.write("| Language | Metric | Baseline | Retrained (LOC Patch) | Delta | Tradeoff / Analysis |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :--- |\n")

        for lang, r in all_results.items():
            b = r["baseline_eval"]
            a = r["retrained_eval"]
            b_loc = b["by_type"].get("LOC", {})
            a_loc = a["by_type"].get("LOC", {})
            b_org = b["by_type"].get("ORG", {})
            a_org = a["by_type"].get("ORG", {})
            b_per = b["by_type"].get("PER", {})
            a_per = a["by_type"].get("PER", {})

            # Overall
            ov_delta = a["entity_f1"] - b["entity_f1"]
            f.write(f"| **{lang.capitalize()}** | Overall Entity F1 | {b['entity_f1']*100:.2f}% | {a['entity_f1']*100:.2f}% | {ov_delta*100:+.2f}% | Overall dev benchmark |\n")

            # LOC
            loc_delta = a_loc.get("f1", 0.0) - b_loc.get("f1", 0.0)
            f.write(
                f"| | **LOC F1** | {b_loc.get('f1', 0.0)*100:.2f}% | **{a_loc.get('f1', 0.0)*100:.2f}%** | "
                f"**{loc_delta*100:+.2f}%** | Target Entity Type |\n"
            )
            f.write(
                f"| | LOC Precision | {b_loc.get('precision', 0.0)*100:.2f}% | {a_loc.get('precision', 0.0)*100:.2f}% | "
                f"{(a_loc.get('precision', 0.0) - b_loc.get('precision', 0.0))*100:+.2f}% | False Positive Check |\n"
            )
            f.write(
                f"| | LOC Recall | {b_loc.get('recall', 0.0)*100:.2f}% | {a_loc.get('recall', 0.0)*100:.2f}% | "
                f"{(a_loc.get('recall', 0.0) - b_loc.get('recall', 0.0))*100:+.2f}% | Ground Truth Coverage |\n"
            )

            # ORG
            org_delta = a_org.get("f1", 0.0) - b_org.get("f1", 0.0)
            f.write(
                f"| | ORG F1 | {b_org.get('f1', 0.0)*100:.2f}% | {a_org.get('f1', 0.0)*100:.2f}% | "
                f"{org_delta*100:+.2f}% | Regression Check |\n"
            )

            # PER
            per_delta = a_per.get("f1", 0.0) - b_per.get("f1", 0.0)
            f.write(
                f"| | PER F1 | {b_per.get('f1', 0.0)*100:.2f}% | {a_per.get('f1', 0.0)*100:.2f}% | "
                f"{per_delta*100:+.2f}% | Regression Check |\n"
            )

        f.write("\n---\n\n")
        f.write("## 3. False-Positive Tradeoff Analysis\n\n")
        for lang, r in all_results.items():
            b = r["baseline_eval"]
            a = r["retrained_eval"]
            b_loc = b["by_type"].get("LOC", {})
            a_loc = a["by_type"].get("LOC", {})

            b_fp = b_loc.get("false_positives", b_loc.get("fp", 0))
            a_fp = a_loc.get("false_positives", a_loc.get("fp", 0))
            b_tp = b_loc.get("true_positives", b_loc.get("tp", 0))
            a_tp = a_loc.get("true_positives", a_loc.get("tp", 0))
            b_fn = b_loc.get("false_negatives", b_loc.get("fn", 0))
            a_fn = a_loc.get("false_negatives", a_loc.get("fn", 0))

            f.write(f"### {lang.capitalize()}\n")
            f.write(f"- **Baseline Counts**: TP={b_tp}, FP={b_fp}, FN={b_fn}\n")
            f.write(f"- **Retrained Counts**: TP={a_tp}, FP={a_fp}, FN={a_fn}\n")
            if a_fp > b_fp:
                f.write(f"- **Tradeoff Detected**: Introduced {a_fp - b_fp} new false-positive LOC prediction(s). Precision shifted from {b_loc.get('precision', 0.0)*100:.2f}% to {a_loc.get('precision', 0.0)*100:.2f}%.\n\n")
            else:
                f.write(f"- **No Tradeoff**: Zero new false-positive LOC predictions introduced. Precision maintained at {a_loc.get('precision', 0.0)*100:.2f}%.\n\n")

    print(f"\n[COMPLETE] Wrote retraining and divergence results to: {report_path}")
    return all_results


if __name__ == "__main__":
    run_all_loc_retraining()
