"""
Inventory and Loadability Assessment for all checkpoints in checkpoints/.

Tests:
1. File existence, size, SHA-256 hash.
2. Architecture matching against current production model (NERMultilingualModel).
3. Forward pass on dummy input for loadable checkpoints.
4. Detailed documentation of legacy checkpoints that predate language conditioning / multilingual vocab.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from model.ner_model_v4_multilingual import NERMultilingualModel


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_checkpoint_loadability(
    ckpt_path: Path,
    word_vocab_size: int = 89196,
    char_vocab_size: int = 13022,
) -> dict[str, Any]:
    file_size = ckpt_path.stat().st_size
    file_hash = compute_sha256(ckpt_path)

    try:
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = data.get("model_state_dict", data)
    except Exception as e:
        return {
            "name": ckpt_path.name,
            "size_bytes": file_size,
            "sha256": file_hash,
            "loadable": False,
            "reason": f"Corrupted or unreadable torch file: {e}",
        }

    # Analyze state dict keys
    keys = list(sd.keys())
    has_lang_cond = any("language_conditioning" in k for k in keys)
    has_char_cnn = any("char_cnn" in k for k in keys)
    has_attention = any("attention" in k for k in keys)
    has_crf = any("crf" in k for k in keys)

    # Word embedding vocab size in checkpoint
    w_shape = sd.get("word_embedding.embedding.weight", torch.empty(0)).shape
    c_shape = sd.get("char_cnn.char_embedding.weight", torch.empty(0)).shape

    # Determine Phase of origin based on name and architecture
    name = ckpt_path.name
    origin_phase = "Unknown"
    if "baseline_" in name:
        origin_phase = "Phase 5 (Baseline BiLSTM+CRF)"
    elif "charcnn_" in name:
        origin_phase = "Phase 6 (CharCNN+BiLSTM+CRF)"
    elif "attention_" in name:
        origin_phase = "Phase 7 (Self-Attention+CharCNN+BiLSTM+CRF)"
    elif "joint_" in name:
        origin_phase = "Phase 8 (Joint Multilingual Pretraining)"
    elif "finetuned_" in name or "fromscratch_" in name:
        origin_phase = "Phase 9 (Language-Specific Fine-Tuning)"
    elif "selftrained_" in name:
        origin_phase = "Phase 10 (Self-Training Iterations)"
    elif "augmented_" in name:
        origin_phase = "Phase 11 (Data Augmentation)"
    elif "final_" in name:
        origin_phase = "Phase 12 / 15 (Locked Final Bundles)"
    elif "patched_v2_" in name:
        origin_phase = "Patch v2 (Corrective Retraining)"
    elif "patched_" in name:
        origin_phase = "Phase 10/11 Patch v1 (Initial Patch)"

    # Test loading into NERMultilingualModel
    model = NERMultilingualModel(
        word_vocab_size=word_vocab_size,
        char_vocab_size=char_vocab_size,
        num_tags=9,
        num_languages=3,
        hidden_dim=128,
    )

    load_status = False
    exclusion_reason = None

    try:
        model.load_state_dict(sd, strict=True)
        model.eval()

        # Dummy forward pass: batch_size=1, seq_len=4, max_char_len=6
        dummy_words = torch.randint(0, word_vocab_size, (1, 4))
        dummy_chars = torch.randint(0, char_vocab_size, (1, 4, 6))
        dummy_lang = torch.tensor([0], dtype=torch.long)
        dummy_mask = torch.ones((1, 4), dtype=torch.bool)

        with torch.no_grad():
            paths = model(dummy_words, dummy_chars, lang_ids=dummy_lang, mask=dummy_mask)

        if len(paths) == 1 and len(paths[0]) == 4:
            load_status = True
    except Exception as e:
        exclusion_reason = str(e)

    return {
        "name": ckpt_path.name,
        "size_bytes": file_size,
        "sha256": file_hash,
        "origin_phase": origin_phase,
        "loadable_under_v4_multilingual": load_status,
        "exclusion_reason": exclusion_reason,
        "architecture_details": {
            "has_lang_conditioning": has_lang_cond,
            "has_char_cnn": has_char_cnn,
            "has_attention": has_attention,
            "has_crf": has_crf,
            "word_embedding_shape": list(w_shape),
            "char_embedding_shape": list(c_shape),
        },
    }


def inventory_all_checkpoints() -> list[dict[str, Any]]:
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_files = sorted(list(ckpt_dir.glob("*.pt")))

    inventory = []
    for p in ckpt_files:
        info = test_checkpoint_loadability(p)
        inventory.append(info)
    return inventory


if __name__ == "__main__":
    inv = inventory_all_checkpoints()
    print("=== CHECKPOINT INVENTORY & LOADABILITY REPORT ===")
    loadable_count = sum(1 for x in inv if x["loadable_under_v4_multilingual"])
    print(f"Total Checkpoints: {len(inv)} | Loadable under NERMultilingualModel: {loadable_count} | Excluded: {len(inv) - loadable_count}\n")

    print(f"{'Checkpoint Name':<32} | {'Loadable?':<9} | {'Origin Phase':<36} | {'Reason / Embedding Shape'}")
    print("-" * 120)
    for x in inv:
        status = "YES" if x["loadable_under_v4_multilingual"] else "NO"
        if x["loadable_under_v4_multilingual"]:
            detail = f"Shape {x['architecture_details']['word_embedding_shape']}"
        else:
            reason = x["exclusion_reason"] or "N/A"
            detail = reason.replace("\n", " ")[:45] + "..." if len(reason) > 45 else reason
        print(f"{x['name']:<32} | {status:<9} | {x['origin_phase']:<36} | {detail}")
