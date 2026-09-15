"""
Download and extract clean Wikipedia article text for Konkani, Assamese, and Sinhala.

Pulls directly from official Wikimedia Wikipedia dumps without using any
pretrained models, tokenizers, or external NLP weights.
"""

from __future__ import annotations

import io
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow.parquet as pq

from utils.config_loader import BaseConfig, load_config, parse_cli_args
from utils.logger import ExperimentLogger
from utils.seed import set_global_seed

WIKIPEDIA_DUMP_URLS: dict[str, str] = {
    "konkani": "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.gom/train-00000-of-00001.parquet",
    "assamese": "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.as/train-00000-of-00001.parquet",
    "sinhala": "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.si/train-00000-of-00001.parquet",
}


def clean_wiki_text(raw_text: str) -> str:
    """Clean raw Wikipedia text by stripping section markers and reference tags."""
    if not raw_text:
        return ""
    # Remove section headings like '== References ==', '=== Notes ==='
    text = re.sub(r"={2,}\s*[^=]+\s*={2,}", "\n", raw_text)
    # Remove citation brackets like [1], [2], [citation needed]
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    # Normalize multiple newlines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def download_and_extract_wikipedia(
    lang: str,
    output_dir: Path,
    max_articles: int | None = None,
) -> tuple[int, int, list[str]]:
    """
    Download Wikipedia parquet dump, extract article text, and save clean text.

    Returns:
        tuple of (article_count, line_count, titles_list)
    """
    if lang not in WIKIPEDIA_DUMP_URLS:
        raise ValueError(f"Unknown language '{lang}'. Available: {list(WIKIPEDIA_DUMP_URLS.keys())}")

    url = WIKIPEDIA_DUMP_URLS[lang]
    output_path = output_dir / f"{lang}" / "wikipedia.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{lang.upper()}] Fetching Wikipedia dump from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "NLPResearchCorpusBot/1.0"})
    
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()

    print(f"[{lang.upper()}] Downloaded {len(data) / (1024 * 1024):.2f} MB. Parsing parquet table...")
    table = pq.read_table(io.BytesIO(data))
    num_rows = table.num_rows

    if max_articles is not None and max_articles > 0:
        table = table.slice(0, min(max_articles, num_rows))

    pydict = table.to_pydict()
    titles = pydict.get("title", [])
    texts = pydict.get("text", [])

    total_lines = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for title, raw_body in zip(titles, texts):
            cleaned = clean_wiki_text(raw_body)
            if cleaned:
                out_f.write(cleaned + "\n")
                total_lines += cleaned.count("\n") + 1

    print(f"[{lang.upper()}] Wrote {len(titles)} articles ({total_lines} lines) to {output_path}")
    return len(titles), total_lines, titles


def run_wikipedia_download(config_path: str = "configs/base_config.yaml", overrides: list[str] | None = None) -> dict[str, Any]:
    cfg: BaseConfig = load_config(config_path, overrides=overrides)
    set_global_seed(cfg.seed)

    logger = ExperimentLogger(log_dir=cfg.logging.log_dir, backend=cfg.logging.backend)
    raw_dir = PROJECT_ROOT / "data" / "raw"

    stats: dict[str, Any] = {}
    max_articles = cfg.data_collection.max_articles_per_wiki if cfg.data_collection else None

    for lang in cfg.project.languages:
        try:
            articles, lines, titles = download_and_extract_wikipedia(
                lang=lang,
                output_dir=raw_dir,
                max_articles=max_articles,
            )
            stats[f"{lang}_wiki_articles"] = articles
            stats[f"{lang}_wiki_lines"] = lines
        except Exception as exc:
            print(f"Error downloading Wikipedia for {lang}: {exc}", file=sys.stderr)
            stats[f"{lang}_wiki_error"] = str(exc)

    logger.log_run(
        config_path=config_path,
        metrics=stats,
        dataset_version="v0.1-wikipedia-raw",
    )
    return stats


if __name__ == "__main__":
    cfg_file, cli_overrides = parse_cli_args(sys.argv[1:])
    run_wikipedia_download(cfg_file, cli_overrides)
