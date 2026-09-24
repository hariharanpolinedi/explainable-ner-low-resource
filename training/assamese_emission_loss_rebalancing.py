"""
Assamese Emission-Loss Rebalancing and Candidate Retraining (Cycle 2).

Directly addresses the confirmed root cause from Step 1b:
- The CRF emission head exhibits an extreme static +10 to +13 logit O-bias,
  driving 100% false-negative collapse on high-density long sentences (e.g. Sentence #200).
- Implements class-calibrated emission-loss rebalancing to penalize entity tag omission
  and counteract the runaway O-tag emission gradient.
- Implements training routines for:
  1. Candidate 1: Label-Correction-Only (trained on train_cycle2_corrected.conll)
  2. Candidate 2: Emission-Loss-Rebalancing-Only (trained on train.conll with rebalanced loss)
  3. Candidate 3: Combined (trained on train_cycle2_corrected.conll with rebalanced loss)
- Strictly verifies divergence gate from checkpoints/improved_assamese_v2.pt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.phase5_targeted_interventions import (
    evaluate_model_metrics,
    load_or_create_multilingual_vocabs,
)
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.checkpoint_divergence_gate import save_patched_checkpoint_with_gate
from training.class_weighted_loss import ClassWeightedCRFLoss, compute_inverse_frequency_weights
from training.multilingual_dataset import (
    MultilingualNERDataset,
    collate_fn_multilingual,
)
from training.train_charcnn import ID_TO_TAG, TAG_TO_ID
from utils.seed import set_global_seed


def train_assamese_candidate(
    candidate_name: str,
    train_conll_path: Path,
    use_loss_rebalancing: bool,
    baseline_ckpt_path: Path,
    output_ckpt_path: Path,
    epochs: int = 5,
    lr: float = 5e-5,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Trains an Assamese candidate starting from checkpoints/improved_assamese_v2.pt.
    """
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    gold_dir = PROJECT_ROOT / "data" / "gold" / "assamese"

    print("\n" + "=" * 80)
    print(f"ASSAMESE CANDIDATE: {candidate_name.upper()} (Device: {device})")
    print(f"Starting Baseline Checkpoint: {baseline_ckpt_path.name}")
    print(f"Training Data Split:          {train_conll_path.name}")
    print(f"Loss Rebalancing Active:      {use_loss_rebalancing}")
    print("=" * 80)

    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)

    train_paths = [
        train_conll_path,
        train_conll_path,  # 2x oversampling matching baseline discipline
    ]

    train_ds = MultilingualNERDataset(
        lang_conll_paths={"assamese": train_paths},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )
    dev_path = gold_dir / "dev.conll"

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

    ckpt = torch.load(baseline_ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)

    baseline_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    base_eval = evaluate_model_metrics(model, dev_path, "assamese", word_to_id, char_to_id, device)
    print(f"Baseline Gold-Dev F1: {base_eval['entity_f1']:.2f}% (P: {base_eval['entity_precision']:.2f}%, R: {base_eval['entity_recall']:.2f}%)")

    # Set up loss rebalancing if enabled
    if use_loss_rebalancing:
        # Calibrated weights targeting the +10 logit O-bias gap:
        # O: 1.0, B/I PER: 2.5, B/I LOC: 2.5, B/I ORG: 3.5, B/I MISC: 3.0
        class_weights = compute_inverse_frequency_weights()
        loss_fn = ClassWeightedCRFLoss(class_weights=class_weights).to(device)
        print(f"Emission Loss Rebalancing initialized with class weights: {class_weights.tolist()}")
    else:
        loss_fn = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_cand_state: dict[str, torch.Tensor] | None = None
    best_eval: dict[str, Any] | None = None
    best_f1 = -1.0

    print(f"Retraining on {len(train_ds):,} sentences for {epochs} epochs...")

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

            if use_loss_rebalancing and loss_fn is not None:
                # Custom forward pass with rebalanced emissions
                batch_size_cur, seq_len_cur = word_ids.shape
                word_repr = model.word_embedding(word_ids)
                lang_repr = model.lang_embedding(lang_ids, seq_len_cur)
                char_repr = model.char_cnn(char_ids)
                combined = model.concat_dropout(torch.cat([word_repr, char_repr, lang_repr], dim=-1))
                lstm_hidden = model.encoder(combined, mask=mask, return_hidden=True)
                attended_states, _ = model.attention(lstm_hidden, mask=mask)
                emissions = model.fc_shared(attended_states)

                loss = loss_fn(model.crf_shared, emissions, tags, mask=mask)
            else:
                loss = model(word_ids, char_ids, lang_ids=lang_ids, tags=tags, mask=mask)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        ep_eval = evaluate_model_metrics(model, dev_path, "assamese", word_to_id, char_to_id, device)
        cur_f1 = ep_eval["entity_f1"]
        print(f"  Epoch {ep:2d}/{epochs:2d} - Loss: {avg_loss:.4f} | Dev F1: {cur_f1:.2f}% (P: {ep_eval['entity_precision']:.2f}%, R: {ep_eval['entity_recall']:.2f}%)")

        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_eval = ep_eval
            best_cand_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_cand_state is not None

    metadata = {
        "candidate": candidate_name,
        "epochs_trained": epochs,
        "best_dev_f1": best_f1,
        "dev_eval": best_eval,
        "use_loss_rebalancing": use_loss_rebalancing,
        "train_split": train_conll_path.name,
        "seed": seed,
    }

    passed, div = save_patched_checkpoint_with_gate(
        candidate_state=best_cand_state,
        baseline_state=baseline_state,
        output_path=output_ckpt_path,
        metadata=metadata,
        min_pct_changed=1.0,
        min_l2_diff=0.01,
    )

    if not passed:
        raise RuntimeError(f"Divergence gate FAILED for {output_ckpt_path.name}: {div}")

    return {
        "candidate": candidate_name,
        "best_dev_f1": best_f1,
        "dev_eval": best_eval,
        "divergence": div,
        "checkpoint_path": str(output_ckpt_path),
    }


if __name__ == "__main__":
    base_ckpt = PROJECT_ROOT / "checkpoints" / "improved_assamese_v2.pt"
    orig_train = PROJECT_ROOT / "data" / "gold" / "assamese" / "train.conll"
    corr_train = PROJECT_ROOT / "data" / "gold" / "assamese" / "train_cycle2_corrected.conll"

    # Candidate 1: Label-Correction-Only
    out_c1 = PROJECT_ROOT / "checkpoints" / "cycle2_assamese_label_corrected.pt"
    train_assamese_candidate("cycle2_assamese_label_corrected", corr_train, False, base_ckpt, out_c1, epochs=5, lr=5e-5)

    # Candidate 2: Loss-Rebalancing-Only
    out_c2 = PROJECT_ROOT / "checkpoints" / "cycle2_assamese_loss_rebalanced.pt"
    train_assamese_candidate("cycle2_assamese_loss_rebalanced", orig_train, True, base_ckpt, out_c2, epochs=5, lr=5e-5)

    # Candidate 3: Combined
    out_c3 = PROJECT_ROOT / "checkpoints" / "cycle2_assamese_combined.pt"
    train_assamese_candidate("cycle2_assamese_combined", corr_train, True, base_ckpt, out_c3, epochs=5, lr=5e-5)
