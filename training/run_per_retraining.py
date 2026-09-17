"""
PER (Person-Name) Retraining Pipeline with Hard Divergence Gate.

Retrains Konkani, Sinhala, and Assamese models branching from their genuine-best checkpoints:
- Konkani: checkpoints/selftrained_konkani_best.pt
- Sinhala: checkpoints/conjunct_fixed_sinhala.pt
- Assamese: checkpoints/augmented_assamese.pt

Procedure per language:
1. Pools Gold Train (oversampled 2x) + PER Augmentation (+ existing conjunct robustness for Sinhala).
2. Evaluates on untouched locked Gold-Dev (data/gold/{lang}/dev.conll). Test split remains strictly locked.
3. Computes parameter divergence via training/checkpoint_divergence_gate.py (threshold: >=1.0% changed, L2 >= 0.01).
4. Saves verified checkpoints: checkpoints/per_fixed_{lang}.pt.
5. Updates deployment bundles (deployment/bundles/{lang}/v1/model.pt) if gate passed and dev F1 is preserved.
6. Emits comprehensive report: reports/per_expansion_retraining_results.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.compare_joint_vs_monolingual import evaluate_joint_model_on_language
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.checkpoint_divergence_gate import (
    save_patched_checkpoint_with_gate,
    verify_weight_divergence,
)
from training.multilingual_dataset import (
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from utils.seed import set_global_seed

CONFIGS = {
    "konkani": {
        "baseline_ckpt": PROJECT_ROOT / "checkpoints" / "selftrained_konkani_best.pt",
        "output_ckpt": PROJECT_ROOT / "checkpoints" / "per_fixed_konkani.pt",
        "extra_train_files": [PROJECT_ROOT / "data" / "augmented" / "konkani" / "per_augmentation.conll"],
    },
    "sinhala": {
        "baseline_ckpt": PROJECT_ROOT / "checkpoints" / "conjunct_fixed_sinhala.pt",
        "output_ckpt": PROJECT_ROOT / "checkpoints" / "per_fixed_sinhala.pt",
        "extra_train_files": [
            PROJECT_ROOT / "data" / "augmented" / "sinhala" / "conjunct_robustness.conll",
            PROJECT_ROOT / "data" / "augmented" / "sinhala" / "per_augmentation.conll",
        ],
    },
    "assamese": {
        "baseline_ckpt": PROJECT_ROOT / "checkpoints" / "augmented_assamese.pt",
        "output_ckpt": PROJECT_ROOT / "checkpoints" / "per_fixed_assamese.pt",
        "extra_train_files": [PROJECT_ROOT / "data" / "augmented" / "assamese" / "per_augmentation.conll"],
    },
}


def train_single_language_per(
    lang: str,
    epochs: int = 5,
    lr: float = 5e-5,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    gold_dir = PROJECT_ROOT / "data" / "gold" / lang

    cfg = CONFIGS[lang]
    base_ckpt_path = cfg["baseline_ckpt"]
    out_ckpt_path = cfg["output_ckpt"]

    print("\n" + "=" * 80)
    print(f"[{lang.upper()}] PER-EXPANSION RETRAINING (Device: {device})")
    print(f"Starting Baseline Checkpoint: {base_ckpt_path.name}")
    print("=" * 80)

    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)

    # Training paths: Gold Train (oversampled 2x) + PER augmentation
    train_paths = [gold_dir / "train.conll", gold_dir / "train.conll"] + cfg["extra_train_files"]
    train_ds = MultilingualNERDataset(
        lang_conll_paths={lang: train_paths},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )
    dev_ds = MultilingualNERDataset(
        lang_conll_paths={lang: [gold_dir / "dev.conll"]},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_multilingual,
    )

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
    ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)

    baseline_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate untouched baseline on gold-dev
    baseline_eval = evaluate_joint_model_on_language(model, dev_ds, device=device)
    base_f1 = baseline_eval["entity_f1"]
    print(
        f"[{lang.upper()}] Baseline Gold-Dev - Overall F1: {base_f1*100:.2f}% | "
        f"PER: {baseline_eval['by_type'].get('PER', {}).get('f1', 0.0)*100:.2f}% | "
        f"LOC: {baseline_eval['by_type'].get('LOC', {}).get('f1', 0.0)*100:.2f}% | "
        f"ORG: {baseline_eval['by_type'].get('ORG', {}).get('f1', 0.0)*100:.2f}%"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_cand_state: dict[str, torch.Tensor] | None = None
    best_eval: dict[str, Any] | None = None
    best_f1 = -1.0

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
        ep_f1 = ep_eval["entity_f1"]

        print(
            f"  Epoch {ep:2d}/{epochs:2d} - Loss: {avg_loss:.4f} | "
            f"Dev F1: {ep_f1*100:.2f}% | "
            f"PER: {ep_eval['by_type'].get('PER', {}).get('f1', 0.0)*100:.2f}% | "
            f"LOC: {ep_eval['by_type'].get('LOC', {}).get('f1', 0.0)*100:.2f}% | "
            f"ORG: {ep_eval['by_type'].get('ORG', {}).get('f1', 0.0)*100:.2f}%"
        )

        if ep_f1 >= best_f1:
            best_f1 = ep_f1
            best_eval = ep_eval
            best_cand_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_cand_state is not None and best_eval is not None

    # Verify Divergence Gate
    passed, div, div_msg = verify_weight_divergence(
        best_cand_state,
        baseline_state,
        min_pct_changed=1.0,
        min_l2_diff=0.01,
    )

    if passed:
        metadata = {
            "language": lang,
            "training_phase": "Phase 12 PER Coverage Expansion Retraining",
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "baseline_checkpoint": str(base_ckpt_path.name),
            "baseline_gold_dev_f1": base_f1,
            "retrained_gold_dev_f1": best_f1,
            "retrained_eval": best_eval,
        }
        save_patched_checkpoint_with_gate(
            candidate_state=best_cand_state,
            baseline_state=baseline_state,
            output_path=out_ckpt_path,
            metadata=metadata,
            min_pct_changed=1.0,
            min_l2_diff=0.01,
        )

        # Update deployed bundle if dev F1 did not regress significantly (> -0.01)
        if best_eval["entity_f1"] >= baseline_eval["entity_f1"] - 0.01:
            bundle_model_path = PROJECT_ROOT / "deployment" / "bundles" / lang / "v1" / "model.pt"
            torch.save(ckpt | {"model_state_dict": best_cand_state}, bundle_model_path)
            print(f"[{lang.upper()} DEPLOYMENT] Updated production bundle model at: {bundle_model_path}")
        else:
            print(f"[{lang.upper()} DEPLOYMENT] Retrained F1 regressed beyond tolerance; retained existing bundle.")
    else:
        print(f"\n[{lang.upper()} DIVERGENCE GATE BLOCKED] {div_msg}")

    return {
        "language": lang,
        "baseline_eval": baseline_eval,
        "retrained_eval": best_eval,
        "divergence_gate_passed": passed,
        "divergence_metrics": div,
        "output_checkpoint": str(out_ckpt_path),
    }


def run_all_per_retraining() -> dict[str, Any]:
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "per_expansion_retraining_results.md"

    results = {}
    for lang in ["konkani", "sinhala", "assamese"]:
        res = train_single_language_per(lang)
        results[lang] = res

    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# PER Coverage Expansion Retraining Results\n\n")
        f.write("**Date**: 2026-09-17  \n")
        f.write("**Auditor**: Senior ML/NLP Engineer  \n")
        f.write("**Evaluation Split**: Locked Gold-Dev ONLY (`data/gold/{lang}/dev.conll`)\n\n")
        f.write("---\n\n")

        f.write("## 1. Divergence Gate Verification Across Languages\n\n")
        f.write("| Language | Gate Status | Parameters Changed | L2 Norm Diff | Max Diff | Saved Checkpoint |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :--- |\n")
        for lang, r in results.items():
            div = r["divergence_metrics"]
            status = "**PASSED**" if r["divergence_gate_passed"] else "**FAILED**"
            f.write(f"| **{lang.title()}** | {status} | {div['pct_changed']:.2f}% | {div['l2_diff']:.4f} | {div['max_diff']:.4f} | `checkpoints/per_fixed_{lang}.pt` |\n")

        f.write("\n---\n\n")
        f.write("## 2. Before vs After Performance on Locked Gold-Dev\n\n")
        for lang, r in results.items():
            b = r["baseline_eval"]
            a = r["retrained_eval"]
            f.write(f"### {lang.title()}\n\n")
            f.write("| Metric | Baseline | Retrained (`per_fixed`) | Delta |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            ov_delta = a["entity_f1"] - b["entity_f1"]
            f.write(f"| **Overall Entity F1** | **{b['entity_f1']*100:.2f}%** | **{a['entity_f1']*100:.2f}%** | **{ov_delta*100:+.2f}%** |\n")
            f.write(f"| Token Accuracy | {b['token_accuracy']*100:.2f}% | {a['token_accuracy']*100:.2f}% | {(a['token_accuracy']-b['token_accuracy'])*100:+.2f}% |\n")
            for etype in ["PER", "LOC", "ORG", "MISC"]:
                bf = b["by_type"].get(etype, {}).get("f1", 0.0)
                af = a["by_type"].get(etype, {}).get("f1", 0.0)
                f.write(f"| {etype} F1 | {bf*100:.2f}% | {af*100:.2f}% | {(af-bf)*100:+.2f}% |\n")
            f.write("\n")

    print(f"\n[COMPLETE] Wrote full PER retraining report to: {report_path}")
    return results


if __name__ == "__main__":
    run_all_per_retraining()
