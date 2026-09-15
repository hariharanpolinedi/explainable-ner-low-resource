"""
Deduplication and quality filtering pipeline for raw multilingual corpora.

Performs:
1. Hash-based sentence deduplication (SHA-256) across sources per language.
2. Script-purity filtering (default >= 90% target script characters).
3. Minimum sentence length enforcement (default >= 3 whitespace tokens).
4. Boilerplate / navigation artifact removal.

Strictly rule-based and hash-based; zero pretrained models or tokenizers used.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import BaseConfig, load_config, parse_cli_args
from utils.logger import ExperimentLogger
from utils.seed import set_global_seed

# Unicode ranges for target language scripts
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "konkani": [(0x0900, 0x097F)],  # Devanagari
    "assamese": [(0x0980, 0x09FF)],  # Bengali-Assamese
    "sinhala": [(0x0D80, 0x0DFF)],  # Sinhala
}

# Boilerplate patterns to drop
BOILERPLATE_PATTERNS = [
    re.compile(r"^https?://", re.IGNORECASE),
    re.compile(r"^(Category|वर्ग|শ্ৰেণী|ප්‍රවර්ගය):", re.IGNORECASE),
    re.compile(r"^(See also|References|External links|इहें बघा|বাহ্যিক সংযোগ|බාහිර සබැඳි)", re.IGNORECASE),
    re.compile(r"^\d+$"),  # Pure digits
]


def is_target_script_char(char: str, script_ranges: list[tuple[int, int]]) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in script_ranges)


def compute_script_purity(text: str, script_ranges: list[tuple[int, int]]) -> float:
    """Calculate ratio of target script alphabetic characters to total alphabetic characters."""
    total_letters = 0
    target_letters = 0

    for ch in text:
        cat = unicodedata.category(ch)
        # Check if character is a Letter (L) or Mark (M)
        if cat.startswith("L") or cat.startswith("M"):
            total_letters += 1
            if is_target_script_char(ch, script_ranges):
                target_letters += 1

    if total_letters == 0:
        return 0.0
    return target_letters / total_letters


def compute_sentence_hash(sentence: str, method: str = "sha256") -> str:
    """Compute hash of normalized sentence for exact deduplication."""
    norm = re.sub(r"\s+", " ", sentence).strip()
    if method == "sha256":
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def filter_and_dedup_file(
    file_path: Path,
    script_ranges: list[tuple[int, int]],
    min_tokens: int,
    purity_threshold: float,
    seen_hashes: set[str],
    dedup_method: str = "sha256",
) -> tuple[int, int]:
    """
    Filter and deduplicate lines in a text file.

    Returns:
        tuple of (original_count, retained_count)
    """
    if not file_path.is_file():
        return 0, 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    original_count = len(raw_lines)
    retained_lines: list[str] = []

    for line in raw_lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Check boilerplate
        if any(pat.search(line_clean) for pat in BOILERPLATE_PATTERNS):
            continue

        # Check token length
        tokens = line_clean.split()
        if len(tokens) < min_tokens:
            continue

        # Check script purity
        purity = compute_script_purity(line_clean, script_ranges)
        if purity < purity_threshold:
            continue

        # Check deduplication
        h = compute_sentence_hash(line_clean, method=dedup_method)
        if h in seen_hashes:
            continue

        seen_hashes.add(h)
        retained_lines.append(line_clean)

    # Overwrite file with clean, deduplicated lines
    with open(file_path, "w", encoding="utf-8") as f:
        for r_line in retained_lines:
            f.write(r_line + "\n")

    return original_count, len(retained_lines)


def process_language(
    lang: str,
    raw_dir: Path,
    min_tokens: int = 3,
    purity_threshold: float = 0.90,
    dedup_method: str = "sha256",
) -> dict[str, int]:
    """Process all raw files for a given language."""
    lang_dir = raw_dir / lang
    if not lang_dir.is_dir():
        return {}

    script_ranges = SCRIPT_RANGES.get(lang, [])
    if not script_ranges:
        raise ValueError(f"No script ranges configured for language '{lang}'")

    seen_hashes: set[str] = set()
    stats: dict[str, int] = {
        "raw_lines": 0,
        "retained_lines": 0,
    }

    # Order of processing: wikipedia, news, other
    for filename in ["wikipedia.txt", "news.txt", "other.txt"]:
        f_path = lang_dir / filename
        orig, ret = filter_and_dedup_file(
            file_path=f_path,
            script_ranges=script_ranges,
            min_tokens=min_tokens,
            purity_threshold=purity_threshold,
            seen_hashes=seen_hashes,
            dedup_method=dedup_method,
        )
        stats[f"{filename}_raw"] = orig
        stats[f"{filename}_retained"] = ret
        stats["raw_lines"] += orig
        stats["retained_lines"] += ret

    print(
        f"[{lang.upper()}] Raw lines: {stats['raw_lines']} -> "
        f"Retained lines: {stats['retained_lines']} "
        f"(Filtered out: {stats['raw_lines'] - stats['retained_lines']})"
    )
    return stats


def run_dedup_and_filter(config_path: str = "configs/base_config.yaml", overrides: list[str] | None = None) -> dict[str, Any]:
    cfg: BaseConfig = load_config(config_path, overrides=overrides)
    set_global_seed(cfg.seed)

    logger = ExperimentLogger(log_dir=cfg.logging.log_dir, backend=cfg.logging.backend)
    raw_dir = PROJECT_ROOT / "data" / "raw"

    min_tokens = cfg.data_collection.min_sentence_length if cfg.data_collection else 3
    purity = cfg.data_collection.script_purity_threshold if cfg.data_collection else 0.90
    method = cfg.data_collection.dedup_method if cfg.data_collection else "sha256"

    all_stats: dict[str, Any] = {}
    for lang in cfg.project.languages:
        stats = process_language(
            lang=lang,
            raw_dir=raw_dir,
            min_tokens=min_tokens,
            purity_threshold=purity,
            dedup_method=method,
        )
        all_stats[f"{lang}_filter_stats"] = stats

    logger.log_run(
        config_path=config_path,
        metrics=all_stats,
        dataset_version="v0.1-filtered-raw",
    )
    return all_stats


if __name__ == "__main__":
    cfg_file, cli_overrides = parse_cli_args(sys.argv[1:])
    run_dedup_and_filter(cfg_file, cli_overrides)
