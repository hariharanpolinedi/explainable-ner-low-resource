"""
Neural Weight Fine-Tuning Pipeline for True Contextual Generalization in Kokborok.
Optimizes the BiLSTM, Char-CNN, Self-Attention, and CRF weights via backpropagation
on the augmented contextual dataset on the GPU.
"""

import sys
import copy
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import MultilingualNERDataset, collate_fn_multilingual
from evaluation.seqeval_metrics import compute_seqeval_metrics
from training.train_charcnn import ID_TO_TAG, TAG_TO_ID
from utils.seed import set_global_seed


def main():
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Neural Weight Fine-Tuning on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    bundle_dir = PROJECT_ROOT / "deployment" / "bundles" / "kokborok" / "v1"
    w_vocab_path = bundle_dir / "word_vocab.json"
    c_vocab_path = bundle_dir / "char_vocab.json"

    with open(w_vocab_path, "r", encoding="utf-8") as f:
        w_data = json.load(f)
        word_to_id = w_data.get("token_to_id", w_data)

    with open(c_vocab_path, "r", encoding="utf-8") as f:
        c_data = json.load(f)
        char_to_id = c_data.get("token_to_id", c_data)

    print(f"Loaded word vocab ({len(word_to_id):,} tokens), char vocab ({len(char_to_id):,} chars)")

    # Initialize model
    model = NERMultilingualModel(
        word_vocab_size=len(word_to_id),
        char_vocab_size=len(char_to_id),
        num_tags=9,
        num_languages=4,
        crf_config="per_language",
        hidden_dim=128,
    ).to(device)

    # Load initial weights
    model_path = bundle_dir / "model.pt"
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    print(f"Successfully loaded base checkpoint from {model_path}")

    # Datasets
    train_file = PROJECT_ROOT / "data" / "processed" / "kokborok_neural_learning_train.conll"
    dev_file = PROJECT_ROOT / "data" / "gold" / "kokborok" / "dev.conll"

    train_ds = MultilingualNERDataset(
        lang_conll_paths={"kokborok": [train_file]},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )
    dev_ds = MultilingualNERDataset(
        lang_conll_paths={"kokborok": [dev_file]},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn_multilingual)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(dev_ds)}")

    # Unfreeze all encoder, char-cnn, and Kokborok CRF parameters for end-to-end learning
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-4)

    max_epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=5e-5)
    best_dev_f1 = -1.0
    best_state = None
    best_metrics = None

    print(f"\nBeginning Neural Training Loop ({max_epochs} epochs)...")
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            w = batch["word_ids"].to(device)
            c = batch["char_ids"].to(device)
            l_ids = batch["lang_ids"].to(device)
            t = batch["tags"].to(device)
            m = batch["mask"].to(device)

            loss = model(w, c, lang_ids=l_ids, tags=t, mask=m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation pass
        model.eval()
        all_gold = []
        all_pred = []
        with torch.no_grad():
            for batch in dev_loader:
                w = batch["word_ids"].to(device)
                c = batch["char_ids"].to(device)
                l_ids = batch["lang_ids"].to(device)
                t = batch["tags"].to(device)
                m = batch["mask"].to(device)
                seq_lengths = batch["seq_lengths"]

                paths = model(w, c, lang_ids=l_ids, mask=m)
                for i, slen in enumerate(seq_lengths):
                    all_gold.append([ID_TO_TAG[tid.item()] for tid in t[i, :slen]])
                    all_pred.append([ID_TO_TAG[tid] for tid in paths[i][:slen]])

        dev_metrics = compute_seqeval_metrics(all_gold, all_pred)
        dev_f1 = dev_metrics["overall"]["entity_f1"] * 100.0
        dev_p = dev_metrics["overall"]["entity_precision"] * 100.0
        dev_r = dev_metrics["overall"]["entity_recall"] * 100.0
        dev_acc = dev_metrics["overall"]["token_accuracy"] * 100.0

        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"  [Epoch {epoch:2d}/{max_epochs}] Loss: {avg_loss:.4f} (lr: {current_lr:.1e}) | "
            f"Gold Dev F1: {dev_f1:5.2f}% (P: {dev_p:5.2f}%, R: {dev_r:5.2f}%, Token Acc: {dev_acc:5.2f}%)"
        )

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dev_metrics

    # Save fine-tuned weights
    save_ckpt_path = PROJECT_ROOT / "checkpoints" / "final_kokborok_neural_learned.pt"
    final_ckpt_path = PROJECT_ROOT / "checkpoints" / "final_kokborok.pt"
    torch.save({"model_state_dict": best_state, "dev_metrics": best_metrics}, save_ckpt_path)
    torch.save({"model_state_dict": best_state, "dev_metrics": best_metrics}, final_ckpt_path)
    print(f"\n[SAVED CHECKPOINT] {save_ckpt_path}")
    print(f"[SAVED FINAL CHECKPOINT] {final_ckpt_path}")

    # Also deploy to active bundle
    torch.save({"model_state_dict": best_state}, model_path)
    print(f"[DEPLOYED TO BUNDLE] {model_path}")
    print(f"Neural network fine-tuning complete! Best Gold Dev F1: {best_dev_f1:.2f}%")


if __name__ == "__main__":
    main()
