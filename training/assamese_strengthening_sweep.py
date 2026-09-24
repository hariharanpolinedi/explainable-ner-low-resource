"""
Assamese Intervention Strengthening Sweep (Cycle 2 — Step 2b).

Systematically sweeps emission loss rebalancing weights (2x, 4x, 8x) and tests
genuine data reinforcement (train_cycle2_corrected_v2.conll) against the
Phase 7 baseline (checkpoints/improved_assamese_v2.pt).

Enforces:
1. Systematic, bounded sweep (2x, 4x, 8x).
2. Divergence gate verification per run.
3. Decoded-tag-level check on Sentence #200 (exact predicted sequence, not just logit gap).
4. Full precision-tradeoff tracking across dev set.
"""

from __future__ import annotations

import json
import math
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
from training.class_weighted_loss import ClassWeightedCRFLoss
from training.multilingual_dataset import (
    MultilingualNERDataset,
    collate_fn_multilingual,
)
from training.train_charcnn import ID_TO_TAG, TAG_TO_ID
from utils.seed import set_global_seed
from weak_supervision.estimate_silver_precision import load_conll_data, extract_entities_from_bio
from preprocessing.tokenizer import ScriptTokenizer


def get_scaled_class_weights(multiplier: float) -> torch.Tensor:
    """
    Multiplier applied to non-O entity tags relative to Step 2 base weights:
    Base (1x): O=1.0, PER=2.5, LOC=2.5, ORG=3.5, MISC=3.0
    At multiplier M: tag weight = 1.0 + M * (base_weight - 1.0) or M * base_weight.
    Here we scale the non-O class weights directly: M * base_weight.
    """
    weights = torch.ones(9, dtype=torch.float32)
    for tid, tag in ID_TO_TAG.items():
        if tag == "O":
            weights[tid] = 1.0
        elif "ORG" in tag:
            weights[tid] = 3.5 * multiplier
        elif "MISC" in tag:
            weights[tid] = 3.0 * multiplier
        elif "PER" in tag:
            weights[tid] = 2.5 * multiplier
        elif "LOC" in tag:
            weights[tid] = 2.5 * multiplier
    return weights


def train_assamese_strengthened_candidate(
    candidate_name: str,
    train_conll_path: Path,
    class_weights: torch.Tensor,
    baseline_ckpt_path: Path,
    output_ckpt_path: Path,
    epochs: int = 5,
    lr: float = 5e-5,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    gold_dir = PROJECT_ROOT / "data" / "gold" / "assamese"

    print("\n" + "=" * 80)
    print(f"ASSAMESE STRENGTHENING: {candidate_name.upper()} (Device: {device})")
    print(f"Starting Checkpoint:   {baseline_ckpt_path.name}")
    print(f"Training Data:         {train_conll_path.name}")
    print(f"Class Weights:         {class_weights.tolist()}")
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

    loss_fn = ClassWeightedCRFLoss(class_weights=class_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_cand_state: dict[str, torch.Tensor] | None = None
    best_eval: dict[str, Any] | None = None
    best_f1 = -1.0

    print(f"Training on {len(train_ds):,} sentences for {epochs} epochs...")

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

            batch_size_cur, seq_len_cur = word_ids.shape
            word_repr = model.word_embedding(word_ids)
            lang_repr = model.lang_embedding(lang_ids, seq_len_cur)
            char_repr = model.char_cnn(char_ids)
            combined = model.concat_dropout(torch.cat([word_repr, char_repr, lang_repr], dim=-1))
            lstm_hidden = model.encoder(combined, mask=mask, return_hidden=True)
            attended_states, _ = model.attention(lstm_hidden, mask=mask)
            emissions = model.fc_shared(attended_states)

            loss = loss_fn(model.crf_shared, emissions, tags, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        model.eval()
        dev_eval = evaluate_model_metrics(model, dev_path, "assamese", word_to_id, char_to_id, device)
        cur_f1 = dev_eval["entity_f1"]
        print(f"  Epoch {ep}/{epochs} - Loss: {avg_loss:.4f} | Dev F1: {cur_f1:.2f}% (P: {dev_eval['entity_precision']:.2f}%, R: {dev_eval['entity_recall']:.2f}%)")

        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_eval = dev_eval
            best_cand_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_cand_state is not None, "Training failed to produce candidate state"

    # Enforce divergence gate against baseline
    print(f"\nChecking divergence gate against {baseline_ckpt_path.name}...")
    passed, div_res = save_patched_checkpoint_with_gate(
        candidate_state=best_cand_state,
        baseline_state=baseline_state,
        output_path=output_ckpt_path,
        min_pct_changed=1.0,
        min_l2_diff=0.01,
        metadata={
            "candidate_name": candidate_name,
            "baseline_checkpoint": baseline_ckpt_path.name,
            "train_data": train_conll_path.name,
            "class_weights": class_weights.tolist(),
            "epochs": epochs,
            "lr": lr,
            "best_dev_f1": best_f1,
            "best_dev_metrics": best_eval,
        },
    )

    if not passed:
        raise RuntimeError(f"Divergence gate failed for {candidate_name}!")

    return {
        "candidate_name": candidate_name,
        "checkpoint_path": output_ckpt_path,
        "class_weights": class_weights.tolist(),
        "best_dev_metrics": best_eval,
        "gate_results": div_res,
    }


def evaluate_sentence_200_emissions(
    model: NERMultilingualModel,
    s200_toks: list[str],
    s200_gold: list[str],
    w_map: dict[str, int],
    c_map: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    """
    Extracts exact per-token emission logits, O-bias gap, and decoded tags for Sentence #200.
    """
    tokenizer = ScriptTokenizer("assamese")
    o_tag_id = TAG_TO_ID["O"]
    entity_tag_ids = [v for k, v in TAG_TO_ID.items() if k != "O"]
    full_len = len(s200_toks)

    tok_gs = [tokenizer.tokenize_characters(w) for w in s200_toks]
    max_c = max(len(gs) for gs in tok_gs) if tok_gs else 1

    w_b = torch.zeros((1, full_len), dtype=torch.long)
    c_b = torch.zeros((1, full_len, max_c), dtype=torch.long)
    l_b = torch.tensor([0], dtype=torch.long)  # assamese = 0
    m_b = torch.ones((1, full_len), dtype=torch.bool)

    for j, (w, gs) in enumerate(zip(s200_toks, tok_gs)):
        w_b[0, j] = w_map.get(w, w_map.get("<UNK>", 1))
        for k, g in enumerate(gs):
            c_b[0, j, k] = c_map.get(g, c_map.get("<UNK>", 1))

    model.eval()
    with torch.no_grad():
        w_emb = model.word_embedding(w_b.to(device))
        c_emb = model.char_cnn(c_b.to(device))
        l_emb = model.lang_embedding(l_b.to(device), full_len)
        x_in = model.concat_dropout(torch.cat([w_emb, c_emb, l_emb], dim=-1))
        lstm_hidden = model.encoder(x_in, mask=m_b.to(device), return_hidden=True)
        attended_states, _ = model.attention(lstm_hidden, mask=m_b.to(device))
        emissions = model.fc_shared(attended_states)

        # Decoded tags via Viterbi
        decoded_paths = model.crf_shared.decode(emissions, mask=m_b.to(device))
        decoded_tags = [ID_TO_TAG[tid] for tid in decoded_paths[0]]

        o_logits = emissions[0, :, o_tag_id].cpu().numpy()
        ent_logits = emissions[0, :, entity_tag_ids]
        max_ent_logits, max_ent_ids = ent_logits.max(dim=-1)
        max_ent_logits = max_ent_logits.cpu().numpy()
        best_ent_tags = [ID_TO_TAG[entity_tag_ids[i]] for i in max_ent_ids.cpu().numpy()]
        o_bias_gaps = o_logits - max_ent_logits

    token_details = []
    for i in range(full_len):
        w = s200_toks[i]
        g = s200_gold[i]
        p = decoded_tags[i]
        o_l = float(o_logits[i])
        e_l = float(max_ent_logits[i])
        gap = float(o_bias_gaps[i])
        best_e = best_ent_tags[i]
        token_details.append({
            "idx": i,
            "token": w,
            "gold": g,
            "pred": p,
            "o_logit": o_l,
            "max_ent_logit": e_l,
            "best_entity_tag": best_e,
            "o_bias_gap": gap,
        })

    mean_gap = float(o_bias_gaps.mean())
    token0_gap = float(o_bias_gaps[0])
    token0_o_logit = float(o_logits[0])
    token0_bloc_logit = float(emissions[0, 0, TAG_TO_ID["B-LOC"]].item())

    gold_entity_tokens = sum(1 for g in s200_gold if g != "O")
    pred_entity_tokens = sum(1 for p in decoded_tags if p != "O")
    hits = sum(1 for p, g in zip(decoded_tags, s200_gold) if g != "O" and p == g)
    pred_spans = list(extract_entities_from_bio(decoded_tags))
    gold_spans = list(extract_entities_from_bio(s200_gold))

    return {
        "mean_gap": mean_gap,
        "token0_gap": token0_gap,
        "token0_o_logit": token0_o_logit,
        "token0_bloc_logit": token0_bloc_logit,
        "gold_entity_tokens": gold_entity_tokens,
        "pred_entity_tokens": pred_entity_tokens,
        "hits": hits,
        "decoded_tags": decoded_tags,
        "pred_spans": pred_spans,
        "gold_spans": gold_spans,
        "token_details": token_details,
    }


def run_full_strengthening_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Strengthening Sweep on Device: {device}")

    proc_dir = PROJECT_ROOT / "data" / "processed"
    gold_dir = PROJECT_ROOT / "data" / "gold" / "assamese"
    ckpts_dir = PROJECT_ROOT / "checkpoints"

    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)

    baseline_ckpt = ckpts_dir / "improved_assamese_v2.pt"
    train_c2_path = gold_dir / "train_cycle2_corrected.conll"
    train_c2_v2_path = gold_dir / "train_cycle2_corrected_v2.conll"

    dev_path = gold_dir / "dev.conll"
    stress_path = gold_dir / "long_multientity_eval_v2.conll"

    stress_sents = load_conll_data(stress_path)
    s200_toks, s200_gold = stress_sents[0]

    # Sweep configurations
    configs = [
        {
            "name": "weight_2x",
            "train_data": train_c2_path,
            "weights": get_scaled_class_weights(2.0),
            "output_ckpt": ckpts_dir / "cycle2_assamese_weight2x.pt",
        },
        {
            "name": "weight_4x",
            "train_data": train_c2_path,
            "weights": get_scaled_class_weights(4.0),
            "output_ckpt": ckpts_dir / "cycle2_assamese_weight4x.pt",
        },
        {
            "name": "weight_8x",
            "train_data": train_c2_path,
            "weights": get_scaled_class_weights(8.0),
            "output_ckpt": ckpts_dir / "cycle2_assamese_weight8x.pt",
        },
        {
            "name": "reinforced_data",
            "train_data": train_c2_v2_path,
            "weights": get_scaled_class_weights(1.0),
            "output_ckpt": ckpts_dir / "cycle2_assamese_reinforced_data.pt",
        },
        {
            "name": "reinforced_weight2x",
            "train_data": train_c2_v2_path,
            "weights": get_scaled_class_weights(2.0),
            "output_ckpt": ckpts_dir / "cycle2_assamese_reinforced_weight2x.pt",
        },
    ]

    # Train all configurations
    sweep_results: dict[str, Any] = {}

    for cfg in configs:
        res = train_assamese_strengthened_candidate(
            candidate_name=cfg["name"],
            train_conll_path=cfg["train_data"],
            class_weights=cfg["weights"],
            baseline_ckpt_path=baseline_ckpt,
            output_ckpt_path=cfg["output_ckpt"],
            epochs=5,
            lr=5e-5,
            batch_size=16,
            seed=42,
        )
        sweep_results[cfg["name"]] = res

    # Evaluate baseline and all sweep candidates
    eval_models = {
        "baseline": baseline_ckpt,
        "combined_step2": ckpts_dir / "cycle2_assamese_combined.pt",
    }
    for cfg in configs:
        eval_models[cfg["name"]] = cfg["output_ckpt"]

    evaluation_summary: dict[str, Any] = {}

    for model_name, ckpt_path in eval_models.items():
        print(f"\n--- Detailed Evaluation for {model_name} ({ckpt_path.name}) ---")
        m = NERMultilingualModel(
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

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        m.load_state_dict(sd)

        dev_metrics = evaluate_model_metrics(m, dev_path, "assamese", word_to_id, char_to_id, device)
        stress_metrics = evaluate_model_metrics(m, stress_path, "assamese", word_to_id, char_to_id, device)
        s200_res = evaluate_sentence_200_emissions(m, s200_toks, s200_gold, word_to_id, char_to_id, device)

        print(f"  Dev F1: {dev_metrics['entity_f1']:.2f}% | P: {dev_metrics['entity_precision']:.2f}% | R: {dev_metrics['entity_recall']:.2f}%")
        print(f"  Stress F1: {stress_metrics['entity_f1']:.2f}% | Collapse Rate: {stress_metrics['collapse_rate']:.2f}% ({stress_metrics['collapse_count']}/{stress_metrics['sents_with_entities']})")
        print(f"  Sentence #200: {s200_res['pred_entity_tokens']}/{s200_res['gold_entity_tokens']} entities detected ({s200_res['hits']} hits). Mean Gap: {s200_res['mean_gap']:+.2f}, Token0 Gap: {s200_res['token0_gap']:+.2f}")

        evaluation_summary[model_name] = {
            "checkpoint": ckpt_path.name,
            "dev_metrics": dev_metrics,
            "stress_metrics": stress_metrics,
            "s200_results": s200_res,
        }

    # Save complete evaluation json
    eval_json_path = PROJECT_ROOT / "reports" / "cycle2_assamese_strengthening_sweep_data.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        # Convert non-serializable elements
        serializable = {}
        for k, v in evaluation_summary.items():
            serializable[k] = {
                "checkpoint": v["checkpoint"],
                "dev_metrics": v["dev_metrics"],
                "stress_metrics": v["stress_metrics"],
                "s200_results": {
                    "mean_gap": v["s200_results"]["mean_gap"],
                    "token0_gap": v["s200_results"]["token0_gap"],
                    "token0_o_logit": v["s200_results"]["token0_o_logit"],
                    "token0_bloc_logit": v["s200_results"]["token0_bloc_logit"],
                    "gold_entity_tokens": v["s200_results"]["gold_entity_tokens"],
                    "pred_entity_tokens": v["s200_results"]["pred_entity_tokens"],
                    "hits": v["s200_results"]["hits"],
                    "decoded_tags": v["s200_results"]["decoded_tags"],
                    "pred_spans": v["s200_results"]["pred_spans"],
                    "gold_spans": v["s200_results"]["gold_spans"],
                    "token_details": v["s200_results"]["token_details"],
                },
            }
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nSaved sweep evaluation data to {eval_json_path}")

    return evaluation_summary


if __name__ == "__main__":
    run_full_strengthening_sweep()
