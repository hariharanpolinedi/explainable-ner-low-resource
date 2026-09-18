"""
Re-verification and clean retraining of Self-Attention Ablation Condition (b).

1. Documents exact flaw in original Phase 12 ablation runner:
   - Artificial truncation of training data to max_sentences_per_lang=1000 (87% data discarded).
   - Training for only 4 epochs from scratch (375 optimizer steps for 10.6M params).
   - Random initialization without convergence.
2. Cleanly retrains no-attention ablation models (NERMultilingualModel with use_attention=False,
   use_char_cnn=True, use_crf=True) on full gold+silver training data across all 3 languages.
3. Verifies divergence using training/checkpoint_divergence_gate.py against final_{lang}.pt.
4. Evaluates on locked gold-dev splits (data/gold/{lang}/dev.conll).
5. Saves clean checkpoints to checkpoints/no_attention_ablation_v2_{lang}.pt.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to sys.path
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


def run_no_attention_reverification(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
) -> dict[str, Any]:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    gold_dir = Path("data/gold")
    silver_dir = Path("data/silver")
    proc_dir = Path("data/processed")
    checkpoints_dir = Path("checkpoints")
    languages = ("konkani", "assamese", "sinhala")

    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir, languages=languages)

    print("\n====================================================================")
    print("  DIAGNOSTIC: Comparing Training Data & Epochs (Original vs Corrected)")
    print("====================================================================")
    print("Original Phase 12 ablation (b):")
    print("  - Data pool: max_sentences_per_lang = 1,000 (3,000 total sentences)")
    print("  - Epochs: 4 (approx 93 steps/epoch = 375 total optimization steps)")
    print("  - Pretrained / warm-start: None (random initialization)")
    print("  - Result: Non-convergence / early abort at 58.58% (gom), 72.59% (as), 51.45% (si)")
    print("\nCorrected Phase 12 ablation (b):")
    print("  - Data pool: FULL gold + silver (7,800 sentences per language = 23,400 total sentences)")
    print(f"  - Epochs: {epochs} with validation tracking on locked gold-dev")
    print("  - Architecture: Char-CNN (90d) + BiLSTM (256d) + Linear-Chain CRF (use_attention=False)")

    # Full training dataset
    train_paths = {
        lang: [gold_dir / lang / "train.conll", silver_dir / lang / "train.conll"]
        for lang in languages
    }
    train_dataset = MultilingualNERDataset(
        lang_conll_paths=train_paths,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )
    print(f"Total training sentences loaded: {len(train_dataset)}")

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

    # Instantiate model with use_attention=False
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
        use_char_cnn=True,
        use_attention=False,  # STRICTLY DISABLED
        use_crf=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_dev_f1s: dict[str, float] = {l: 0.0 for l in languages}
    best_state_dicts: dict[str, dict[str, torch.Tensor]] = {}

    print("\n--- Beginning Training of Clean No-Attention Ablation Model ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            w = batch["word_ids"].to(device)
            c = batch["char_ids"].to(device)
            l = batch["lang_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            loss = model(w, c, lang_ids=l, tags=tags, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Evaluate on gold-dev
        model.eval()
        epoch_f1s = {}
        for lang in languages:
            dev_path = {lang: [gold_dir / lang / "dev.conll"]}
            dev_ds = MultilingualNERDataset(dev_path, word_to_id, char_to_id)
            metrics = evaluate_joint_model_on_language(model, dev_ds, device, batch_size=32)
            f1 = metrics["entity_f1"] * 100
            epoch_f1s[lang] = f1
            if f1 > best_dev_f1s[lang]:
                best_dev_f1s[lang] = f1
                best_state_dicts[lang] = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        macro_f1 = sum(epoch_f1s.values()) / 3
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_loss:.4f} | "
              f"Konkani: {epoch_f1s['konkani']:.2f}% | Assamese: {epoch_f1s['assamese']:.2f}% | "
              f"Sinhala: {epoch_f1s['sinhala']:.2f}% | Macro F1: {macro_f1:.2f}%")

    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration:.2f}s.")
    print("Best Observed Dev F1s during training:")
    for lang in languages:
        print(f"  {lang.capitalize()}: {best_dev_f1s[lang]:.2f}%")
    best_macro = sum(best_dev_f1s.values()) / 3
    print(f"  Macro Dev F1: {best_macro:.2f}%")

    # Save checkpoints per language
    saved_checkpoints: dict[str, Path] = {}
    for lang in languages:
        out_path = checkpoints_dir / f"no_attention_ablation_v2_{lang}.pt"
        ckpt_payload = {
            "model_state_dict": best_state_dicts.get(lang, model.state_dict()),
            "dev_f1": best_dev_f1s[lang],
            "ablation": "no_attention",
            "use_attention": False,
            "use_char_cnn": True,
            "use_crf": True,
            "epochs": epochs,
            "seed": seed,
        }
        torch.save(ckpt_payload, out_path)
        saved_checkpoints[lang] = out_path
        print(f"Saved {out_path.name} (Dev F1: {best_dev_f1s[lang]:.2f}%)")

        # Verify divergence gate against final_{lang}.pt
        baseline_path = checkpoints_dir / f"final_{lang}.pt"
        if baseline_path.is_file():
            base_ckpt = torch.load(baseline_path, map_location="cpu", weights_only=False)
            base_state = base_ckpt.get("model_state_dict", base_ckpt)
            # Compare common keys
            passed, div, msg = verify_weight_divergence(
                cand_state,
                base_state,
                min_pct_changed=1.0,
            )
            print(f"  Divergence gate against {baseline_path.name}: "
                  f"Changed params: {div['pct_changed']:.2f}% (Passed: {passed}) | L2 diff: {div['l2_diff']:.4f}")

    # Final detailed re-evaluation of saved checkpoints on gold-dev
    final_eval_results: dict[str, Any] = {}
    for lang in languages:
        cp_path = saved_checkpoints[lang]
        ckpt = torch.load(cp_path, map_location=device, weights_only=False)
        model_eval = NERMultilingualModel(
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
            use_char_cnn=True,
            use_attention=False,
            use_crf=True,
        ).to(device)
        model_eval.load_state_dict(ckpt["model_state_dict"])
        dev_ds = MultilingualNERDataset({lang: [gold_dir / lang / "dev.conll"]}, word_to_id, char_to_id)
        m = evaluate_joint_model_on_language(model_eval, dev_ds, device, batch_size=32)
        final_eval_results[lang] = {
            "entity_f1": round(m["entity_f1"] * 100, 2),
            "precision": round(m["precision"] * 100, 2),
            "recall": round(m["recall"] * 100, 2),
            "token_accuracy": round(m["token_accuracy"] * 100, 2),
        }
        print(f"\nFinal Gold-Dev Eval [{lang.capitalize()}]:")
        print(f"  Entity F1: {final_eval_results[lang]['entity_f1']}%")
        print(f"  Precision: {final_eval_results[lang]['precision']}% | Recall: {final_eval_results[lang]['recall']}%")
        print(f"  Token Acc: {final_eval_results[lang]['token_accuracy']}%")

    macro_corrected_f1 = sum(v["entity_f1"] for v in final_eval_results.values()) / 3
    final_eval_results["macro"] = {
        "entity_f1": round(macro_corrected_f1, 2),
    }

    out_json = Path("reports/experiment_logs/no_attention_reverification_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(final_eval_results, fp, indent=2)

    return final_eval_results


if __name__ == "__main__":
    run_no_attention_reverification()
