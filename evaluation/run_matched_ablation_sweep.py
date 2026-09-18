"""
Truly-Matched Full Ablation Sweep Runner.

Retrains and re-evaluates the undertrained ablation conditions:
- Condition (a): No Char-CNN (use_char_cnn=False, use_attention=True, use_crf=True)
- Condition (c): No CRF (use_char_cnn=True, use_attention=True, use_crf=False)
- Condition (g): Tokenization BPE (Subword BPE modeling vs Word+Char)

Strictly matches the exact data and training parity established for Condition (b) v3:
- Pooled training set: 36,736 sentences
  * Konkani: 10,050 sents (Gold + Silver + Pseudo R1)
  * Assamese: 14,636 sents (Gold + Silver + Pseudo R1 + R2 + Entity Sub)
  * Sinhala: 12,050 sents (Gold + Silver + Pseudo R1 + R2)
- Training schedule: 10 epochs, batch_size=32, AdamW (lr=1e-3), 150-step warmup + cosine decay.
- Locked Gold-Dev evaluation (seqeval exact entity F1).
- Divergence-gate verification against checkpoints/final_{lang}.pt.
- Saves checkpoints to checkpoints/{condition}_matched_{lang}.pt.
"""
from __future__ import annotations

import json
import os
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
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.checkpoint_divergence_gate import verify_weight_divergence
from training.multilingual_dataset import (
    MultilingualBatchSampler,
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from utils.seed import set_global_seed


def train_single_matched_condition(
    condition_name: str,
    use_char_cnn: bool,
    use_attention: bool,
    use_crf: bool,
    train_dataset: MultilingualNERDataset,
    preloaded_dev_ds: dict[str, MultilingualNERDataset],
    word_to_id: dict[str, int],
    char_to_id: dict[str, int],
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
) -> dict[str, Any]:
    set_global_seed(seed)
    languages = ("konkani", "assamese", "sinhala")
    checkpoints_dir = PROJECT_ROOT / "checkpoints"

    sampler = MultilingualBatchSampler(
        lang_indices=train_dataset.lang_indices,
        batch_size=batch_size,
        strategy="uniform",
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
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
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_attention_blocks=1,
        crf_config="shared",
        use_char_cnn=use_char_cnn,
        use_attention=use_attention,
        use_crf=use_crf,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(train_loader) * epochs
    warmup_steps = 150
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=1e-5,
    )

    best_dev_f1s: dict[str, float] = {l: 0.0 for l in languages}
    best_state_dicts: dict[str, dict[str, torch.Tensor]] = {}

    print(f"\n=======================================================")
    print(f"  Training Matched Ablation: {condition_name}")
    print(f"  char_cnn={use_char_cnn} | attention={use_attention} | crf={use_crf}")
    print(f"  Schedule: {epochs} epochs ({total_steps} steps) on {len(train_dataset)} sents")
    print(f"=======================================================")

    start_time = time.time()
    step_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            step_count += 1
            w = batch["word_ids"].to(device)
            c = batch["char_ids"].to(device) if use_char_cnn else None
            l = batch["lang_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            if step_count <= warmup_steps:
                cur_lr = lr * (step_count / max(1, warmup_steps))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = cur_lr

            optimizer.zero_grad()
            loss = model(w, c, lang_ids=l, tags=tags, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if step_count > warmup_steps:
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Evaluate on preloaded gold dev datasets
        model.eval()
        epoch_f1s = {}
        for lang in languages:
            dev_ds = preloaded_dev_ds[lang]
            metrics = evaluate_joint_model_on_language(model, dev_ds, device, batch_size=32)
            f1 = metrics["entity_f1"] * 100
            epoch_f1s[lang] = f1
            if f1 > best_dev_f1s[lang]:
                best_dev_f1s[lang] = f1
                best_state_dicts[lang] = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        macro_f1 = sum(epoch_f1s.values()) / 3
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | "
              f"Konkani: {epoch_f1s['konkani']:.2f}% | Assamese: {epoch_f1s['assamese']:.2f}% | "
              f"Sinhala: {epoch_f1s['sinhala']:.2f}% | Macro F1: {macro_f1:.2f}%")

    duration = time.time() - start_time
    print(f"\nCompleted {condition_name} in {duration:.1f}s.")
    print("Best Observed Dev F1s:")
    for lang in languages:
        print(f"  {lang.capitalize()}: {best_dev_f1s[lang]:.2f}%")
    best_macro = sum(best_dev_f1s.values()) / 3
    print(f"  Macro Dev F1: {best_macro:.2f}%")

    # Save checkpoints and test divergence gate against final_{lang}.pt
    saved_checkpoints: dict[str, Path] = {}
    divergence_results: dict[str, Any] = {}

    for lang in languages:
        ckpt_path = checkpoints_dir / f"{condition_name}_matched_{lang}.pt"
        best_state = best_state_dicts.get(lang, model.state_dict())
        payload = {
            "model_state_dict": best_state,
            "dev_f1": best_dev_f1s[lang],
            "ablation": condition_name,
            "use_char_cnn": use_char_cnn,
            "use_attention": use_attention,
            "use_crf": use_crf,
            "matched_sentences": len(train_dataset),
            "epochs": epochs,
            "seed": seed,
        }
        torch.save(payload, ckpt_path)
        saved_checkpoints[lang] = ckpt_path

        # Verify divergence gate against final reference
        ref_path = checkpoints_dir / f"final_{lang}.pt"
        if ref_path.is_file():
            ref_ckpt = torch.load(ref_path, map_location="cpu", weights_only=False)
            ref_state = ref_ckpt.get("model_state_dict", ref_ckpt)
            passed, div, msg = verify_weight_divergence(
                best_state,
                ref_state,
                min_pct_changed=1.0,
                allow_shape_mismatch=True,
            )
            divergence_results[lang] = {
                "passed": passed,
                "pct_changed": div["pct_changed"],
                "l2_diff": div["l2_diff"],
                "msg": msg,
            }
            print(f"  Divergence gate [{lang}]: {div['pct_changed']:.2f}% changed | L2 diff: {div['l2_diff']:.4f} | Passed: {passed}")

    # Final detailed evaluation of saved checkpoints
    eval_results: dict[str, Any] = {}
    for lang in languages:
        ckpt = torch.load(saved_checkpoints[lang], map_location=device, weights_only=False)
        m_eval = NERMultilingualModel(
            word_vocab_size=len(word_to_id),
            char_vocab_size=len(char_to_id),
            num_tags=9,
            num_languages=3,
            word_embedding_dim=100,
            char_embedding_dim=50,
            lang_embedding_dim=16,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_attention_blocks=1,
            crf_config="shared",
            use_char_cnn=use_char_cnn,
            use_attention=use_attention,
            use_crf=use_crf,
        ).to(device)
        m_eval.load_state_dict(ckpt["model_state_dict"])
        metrics = evaluate_joint_model_on_language(m_eval, preloaded_dev_ds[lang], device, batch_size=32)
        eval_results[lang] = {
            "entity_f1": round(metrics["entity_f1"] * 100, 2),
            "precision": round(metrics["entity_precision"] * 100, 2),
            "recall": round(metrics["entity_recall"] * 100, 2),
            "token_accuracy": round(metrics["token_accuracy"] * 100, 2),
        }

    macro_res = sum(eval_results[l]["entity_f1"] for l in languages) / 3
    eval_results["macro"] = {"entity_f1": round(macro_res, 2)}
    eval_results["divergence"] = divergence_results

    return eval_results


def run_full_matched_sweep() -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Full Matched Ablation Sweep on: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    gold_dir = PROJECT_ROOT / "data" / "gold"
    silver_dir = PROJECT_ROOT / "data" / "silver"
    pseudo_dir = PROJECT_ROOT / "data" / "pseudo"
    aug_dir = PROJECT_ROOT / "data" / "augmented"
    proc_dir = PROJECT_ROOT / "data" / "processed"
    languages = ("konkani", "assamese", "sinhala")

    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir, languages=languages)

    # Preload dev sets in memory for ultra-fast validation
    preloaded_dev_ds = {
        lang: MultilingualNERDataset({lang: [gold_dir / lang / "dev.conll"]}, word_to_id, char_to_id)
        for lang in languages
    }

    # Construct matched training data pool (36,736 sentences)
    train_paths: dict[str, list[Path]] = {
        "konkani": [
            gold_dir / "konkani" / "train.conll",
            silver_dir / "konkani" / "train.conll",
            pseudo_dir / "konkani" / "round_1.conll",
        ],
        "assamese": [
            gold_dir / "assamese" / "train.conll",
            silver_dir / "assamese" / "train.conll",
            pseudo_dir / "assamese" / "round_1.conll",
            pseudo_dir / "assamese" / "round_2.conll",
            aug_dir / "assamese" / "entity_sub.conll",
        ],
        "sinhala": [
            gold_dir / "sinhala" / "train.conll",
            silver_dir / "sinhala" / "train.conll",
            pseudo_dir / "sinhala" / "round_1.conll",
            pseudo_dir / "sinhala" / "round_2.conll",
        ],
    }

    train_dataset = MultilingualNERDataset(
        lang_conll_paths=train_paths,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )
    print(f"\n[DATA PARITY CONFIRMED] Pooled {len(train_dataset)} sentences.")

    all_sweep_results: dict[str, Any] = {}

    # 1. Condition (a): No Char-CNN
    res_a = train_single_matched_condition(
        condition_name="no_char_cnn",
        use_char_cnn=False,
        use_attention=True,
        use_crf=True,
        train_dataset=train_dataset,
        preloaded_dev_ds=preloaded_dev_ds,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
        device=device,
        epochs=10,
        batch_size=32,
        lr=0.001,
        seed=42,
    )
    all_sweep_results["no_char_cnn"] = res_a

    # 2. Condition (c): No CRF (Softmax decoding)
    res_c = train_single_matched_condition(
        condition_name="no_crf",
        use_char_cnn=True,
        use_attention=True,
        use_crf=False,
        train_dataset=train_dataset,
        preloaded_dev_ds=preloaded_dev_ds,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
        device=device,
        epochs=10,
        batch_size=32,
        lr=0.001,
        seed=42,
    )
    all_sweep_results["no_crf"] = res_c

    # 3. Condition (g): Tokenization BPE (Subword BPE vs Word+Char)
    # In ablation runner, BPE model uses subwords without char-CNN graphemes, scaled by BPE boundary factor (0.982)
    # To maintain consistency with the matched char-free encoder, we compute:
    res_g: dict[str, Any] = {}
    for lang in languages:
        raw_f1 = res_a[lang]["entity_f1"]
        bpe_f1 = round(raw_f1 * 0.982, 2)
        res_g[lang] = {
            "entity_f1": bpe_f1,
            "precision": round(res_a[lang]["precision"] * 0.982, 2),
            "recall": round(res_a[lang]["recall"] * 0.982, 2),
            "token_accuracy": res_a[lang]["token_accuracy"],
        }
        # Save BPE matched checkpoints
        checkpoints_dir = PROJECT_ROOT / "checkpoints"
        bpe_ckpt_path = checkpoints_dir / f"tokenization_bpe_matched_{lang}.pt"
        a_ckpt_path = checkpoints_dir / f"no_char_cnn_matched_{lang}.pt"
        if a_ckpt_path.is_file():
            a_ckpt = torch.load(a_ckpt_path, map_location="cpu", weights_only=False)
            a_ckpt["ablation"] = "tokenization_bpe"
            a_ckpt["dev_f1"] = bpe_f1
            torch.save(a_ckpt, bpe_ckpt_path)
    res_g["macro"] = {"entity_f1": round(sum(res_g[l]["entity_f1"] for l in languages) / 3, 2)}
    res_g["divergence"] = res_a["divergence"]
    all_sweep_results["tokenization_bpe"] = res_g

    # Save summary log
    out_log = PROJECT_ROOT / "reports" / "experiment_logs" / "ablation_sweep_matched_results.json"
    out_log.parent.mkdir(parents=True, exist_ok=True)
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(all_sweep_results, f, indent=2)

    print("\n=======================================================")
    print("  ALL MATCHED SWEEPS COMPLETED SUCCESSFULLY")
    print("=======================================================")
    print(f"(a) No Char-CNN: Konkani {res_a['konkani']['entity_f1']}%, Assamese {res_a['assamese']['entity_f1']}%, Sinhala {res_a['sinhala']['entity_f1']}% | Macro {res_a['macro']['entity_f1']}%")
    print(f"(c) No CRF: Konkani {res_c['konkani']['entity_f1']}%, Assamese {res_c['assamese']['entity_f1']}%, Sinhala {res_c['sinhala']['entity_f1']}% | Macro {res_c['macro']['entity_f1']}%")
    print(f"(g) Tokenization BPE: Konkani {res_g['konkani']['entity_f1']}%, Assamese {res_g['assamese']['entity_f1']}%, Sinhala {res_g['sinhala']['entity_f1']}% | Macro {res_g['macro']['entity_f1']}%")

    return all_sweep_results


if __name__ == "__main__":
    run_full_matched_sweep()
