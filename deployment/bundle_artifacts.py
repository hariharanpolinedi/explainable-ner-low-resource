"""
Phase 15: Artifact Bundling Pipeline.

Packages each language's locked Phase-12 final checkpoint, unified vocabularies,
and configuration into a standalone, versioned deployment bundle:
    deployment/bundles/{lang}/{version}/

Verifies cryptographic hashes and embedding-layer dimension compatibility
to guarantee that the deployed inference engine never encounters a vocab/checkpoint mismatch.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

LANGUAGES = ("konkani", "assamese", "sinhala")


def compute_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def bundle_language_artifacts(
    lang: str,
    version: str = "v1",
    base_bundles_dir: Path | str | None = None,
    verify_only: bool = False,
) -> dict[str, Any]:
    """
    Bundle checkpoint, vocabs, and config for a single language with compatibility verification.
    """
    if base_bundles_dir is None:
        base_bundles_dir = PROJECT_ROOT / "deployment" / "bundles"
    else:
        base_bundles_dir = Path(base_bundles_dir)

    bundle_dir = base_bundles_dir / lang / version
    bundle_dir.mkdir(parents=True, exist_ok=True)

    ckpt_src = PROJECT_ROOT / "checkpoints" / f"final_{lang}.pt"
    config_src = PROJECT_ROOT / "configs" / f"final_{lang}.yaml"
    word_vocab_src = PROJECT_ROOT / "data" / "processed" / "multilingual" / "word_vocab.json"
    char_vocab_src = PROJECT_ROOT / "data" / "processed" / "multilingual" / "char_vocab.json"

    for src, name in [
        (ckpt_src, f"checkpoint ({ckpt_src.name})"),
        (config_src, f"config ({config_src.name})"),
        (word_vocab_src, "word vocabulary"),
        (char_vocab_src, "char vocabulary"),
    ]:
        if not src.is_file():
            raise FileNotFoundError(f"Missing required source artifact for {lang}: {name} at {src}")

    # 1. Load Vocabularies
    with open(word_vocab_src, "r", encoding="utf-8") as f:
        word_vocab_data = json.load(f)
    word_to_id = word_vocab_data.get("token_to_id", word_vocab_data)
    word_vocab_size = len(word_to_id)

    with open(char_vocab_src, "r", encoding="utf-8") as f:
        char_vocab_data = json.load(f)
    char_to_id = char_vocab_data.get("token_to_id", char_vocab_data)
    char_vocab_size = len(char_to_id)

    # 2. Inspect Checkpoint & Verify Dimensions
    ckpt = torch.load(ckpt_src, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)

    w_emb_shape = sd["word_embedding.embedding.weight"].shape
    c_emb_shape = sd["char_cnn.char_embedding.weight"].shape

    if w_emb_shape[0] != word_vocab_size:
        raise ValueError(
            f"FATAL COMPATIBILITY MISMATCH for {lang}!\n"
            f"Model word embedding expects vocab size {w_emb_shape[0]}, "
            f"but word_vocab.json contains {word_vocab_size} tokens.\n"
            f"Checkpoint: {ckpt_src}\nVocab: {word_vocab_src}"
        )

    if c_emb_shape[0] != char_vocab_size:
        raise ValueError(
            f"FATAL COMPATIBILITY MISMATCH for {lang}!\n"
            f"Model char embedding expects vocab size {c_emb_shape[0]}, "
            f"but char_vocab.json contains {char_vocab_size} chars.\n"
            f"Checkpoint: {ckpt_src}\nVocab: {char_vocab_src}"
        )

    ckpt_hash = compute_sha256(ckpt_src)
    config_hash = compute_sha256(config_src)
    w_vocab_hash = compute_sha256(word_vocab_src)
    c_vocab_hash = compute_sha256(char_vocab_src)

    print(f"[{lang.upper()}] Compatibility verified:")
    print(f"  Word vocab size: {word_vocab_size} == embedding {w_emb_shape}")
    print(f"  Char vocab size: {char_vocab_size} == embedding {c_emb_shape}")

    if verify_only:
        return {
            "language": lang,
            "status": "VERIFIED_ONLY",
            "word_vocab_size": word_vocab_size,
            "char_vocab_size": char_vocab_size,
        }

    # 3. Copy files into bundle
    dest_model = bundle_dir / "model.pt"
    dest_config = bundle_dir / "config.yaml"
    dest_word_vocab = bundle_dir / "word_vocab.json"
    dest_char_vocab = bundle_dir / "char_vocab.json"

    shutil.copy2(ckpt_src, dest_model)
    shutil.copy2(config_src, dest_config)
    shutil.copy2(word_vocab_src, dest_word_vocab)
    shutil.copy2(char_vocab_src, dest_char_vocab)

    # 4. Generate bundle metadata
    metadata = {
        "language": lang,
        "bundle_version": version,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "VERIFIED_COMPATIBLE",
        "artifacts": {
            "model": {
                "filename": "model.pt",
                "original_source": str(ckpt_src.relative_to(PROJECT_ROOT)),
                "sha256": ckpt_hash,
                "word_vocab_size": word_vocab_size,
                "char_vocab_size": char_vocab_size,
                "word_embedding_dim": int(w_emb_shape[1]),
                "char_embedding_dim": int(c_emb_shape[1]),
            },
            "config": {
                "filename": "config.yaml",
                "original_source": str(config_src.relative_to(PROJECT_ROOT)),
                "sha256": config_hash,
            },
            "word_vocab": {
                "filename": "word_vocab.json",
                "original_source": str(word_vocab_src.relative_to(PROJECT_ROOT)),
                "sha256": w_vocab_hash,
                "vocab_size": word_vocab_size,
            },
            "char_vocab": {
                "filename": "char_vocab.json",
                "original_source": str(char_vocab_src.relative_to(PROJECT_ROOT)),
                "sha256": c_vocab_hash,
                "vocab_size": char_vocab_size,
            },
        },
    }

    metadata_path = bundle_dir / "bundle_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"  Bundle created successfully at: {bundle_dir}")
    return metadata


def create_all_bundles(version: str = "v1", base_bundles_dir: Path | str | None = None) -> dict[str, Any]:
    """Create bundles for all supported languages."""
    print("=" * 80)
    print(f"CREATING VERSIONED DEPLOYMENT BUNDLES ({version})")
    print("=" * 80)

    results = {}
    for lang in LANGUAGES:
        meta = bundle_language_artifacts(lang, version=version, base_bundles_dir=base_bundles_dir)
        results[lang] = meta

    print("=" * 80)
    print("ALL DEPLOYMENT BUNDLES SUCCESSFULLY VERIFIED AND CREATED.")
    print("=" * 80)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Package versioned deployment bundles per language")
    parser.add_argument("--version", type=str, default="v1", help="Bundle version identifier (default: v1)")
    parser.add_argument("--output-dir", type=str, default="deployment/bundles", help="Base directory for bundles")
    parser.add_argument("--verify-only", action="store_true", help="Only verify compatibility without copying")
    args = parser.parse_args()

    if args.verify_only:
        for lang in LANGUAGES:
            bundle_language_artifacts(lang, version=args.version, base_bundles_dir=args.output_dir, verify_only=True)
    else:
        create_all_bundles(version=args.version, base_bundles_dir=args.output_dir)


if __name__ == "__main__":
    main()
