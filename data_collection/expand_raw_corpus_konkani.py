"""
Konkani Raw Corpus Expansion Script.

Sources authentic, high-quality Devanagari Konkani news and public press text from
Bhaangarbhuin (Goa's premier daily Devanagari newspaper, hosting 37,000+ public articles)
to eliminate the 19,020 sentence shortfall and bring Konkani past the Phase-1 50,000-sentence floor.

Strict Phase-1 constraints:
1. Script purity >= 0.90 (Devanagari Unicode block U+0900 - U+097F).
2. Min length >= 3 whitespace tokens.
3. Strict SHA-256 sentence hash deduplication.
4. Zero pretrained models, tokenizers, or external weights.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.normalize import normalize_text
from preprocessing.sentence_segmenter import segment_sentences
from data_collection.corpus_stats import run_corpus_stats

TARGET_NEW_SENTENCES = 20500
PHASE1_FLOOR = 50000


class HTMLTextExtractor(HTMLParser):
    """Clean HTML tags, styles, and scripts while preserving text structure."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []
        self.ignore = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav", "header", "footer", "aside"}:
            self.ignore = True
        elif tag in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.texts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "header", "footer", "aside"}:
            self.ignore = False
        elif tag in {"p", "br", "div", "li"}:
            self.texts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignore:
            self.texts.append(data)

    def get_text(self) -> str:
        raw = "".join(self.texts)
        lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def clean_html_markup(raw_html: str) -> str:
    parser = HTMLTextExtractor()
    parser.feed(raw_html)
    return parser.get_text()


def compute_devanagari_purity(text: str) -> float:
    """Compute ratio of Devanagari characters to all alphabetic characters."""
    letters = 0
    devanagari = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("M"):
            letters += 1
            if 0x0900 <= ord(ch) <= 0x097F:
                devanagari += 1
    if letters == 0:
        return 0.0
    return devanagari / letters


def fetch_posts_page(page_idx: int, per_page: int = 100) -> list[str]:
    """Fetch a page of posts from Bhaangarbhuin WordPress REST API."""
    url = f"https://bhaangarbhuin.com/wp-json/wp/v2/posts?per_page={per_page}&page={page_idx}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NLPResearchCorpusBot/1.0"},
    )
    extracted_lines: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for item in data:
            for field in ["title", "content", "excerpt"]:
                html_chunk = item.get(field, {}).get("rendered", "")
                if html_chunk:
                    plain_text = clean_html_markup(html_chunk)
                    for line in plain_text.splitlines():
                        line_clean = line.strip()
                        if line_clean:
                            extracted_lines.append(line_clean)
    except Exception as exc:
        print(f"[Page {page_idx}] Fetch warning: {exc}")
    return extracted_lines


def load_existing_hashes(raw_konkani_dir: Path) -> set[str]:
    """Accumulate SHA-256 hashes of all existing Konkani lines to ensure 0 duplication."""
    hashes = set()
    for fname in ["wikipedia.txt", "news.txt", "other.txt"]:
        fpath = raw_konkani_dir / fname
        if fpath.is_file():
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    norm = re.sub(r"\s+", " ", line).strip()
                    if norm:
                        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
                        hashes.add(h)
    return hashes


def expand_konkani_corpus(
    target_new_sentences: int = TARGET_NEW_SENTENCES,
    max_workers: int = 5,
) -> dict[str, Any]:
    raw_konkani_dir = PROJECT_ROOT / "data" / "raw" / "konkani"
    news_file = raw_konkani_dir / "news.txt"
    processed_file = PROJECT_ROOT / "data" / "processed" / "konkani" / "sentences.txt"

    print("Checking existing Konkani raw volume...")
    existing_hashes = load_existing_hashes(raw_konkani_dir)
    print(f"Loaded {len(existing_hashes):,} existing unique sentence hashes across raw Konkani files.")

    new_clean_lines: list[str] = []
    seen_new_hashes: set[str] = set()

    page = 1
    batch_size = 10  # fetch 10 pages concurrently per batch
    start_time = time.time()

    print(f"Expanding Konkani raw corpus: target +{target_new_sentences:,} script-pure sentences...")

    while len(new_clean_lines) < target_new_sentences and page < 80:
        pages_to_fetch = list(range(page, page + batch_size))
        print(f"Fetching Bhaangarbhuin news pages {pages_to_fetch[0]}..{pages_to_fetch[-1]}...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {executor.submit(fetch_posts_page, p): p for p in pages_to_fetch}
            for future in concurrent.futures.as_completed(future_to_page):
                p_idx = future_to_page[future]
                try:
                    lines = future.result()
                    for line in lines:
                        # 1. Length filter
                        tokens = line.split()
                        if len(tokens) < 3:
                            continue

                        # 2. Devanagari script purity >= 90%
                        purity = compute_devanagari_purity(line)
                        if purity < 0.90:
                            continue

                        # 3. Exact SHA-256 deduplication
                        norm = re.sub(r"\s+", " ", line).strip()
                        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
                        if h in existing_hashes or h in seen_new_hashes:
                            continue

                        seen_new_hashes.add(h)
                        new_clean_lines.append(line)

                        if len(new_clean_lines) >= target_new_sentences:
                            break
                except Exception as exc:
                    print(f"[Page {p_idx}] Error: {exc}")

        page += batch_size
        elapsed = time.time() - start_time
        print(f"Collected {len(new_clean_lines):,} / {target_new_sentences:,} clean sentences ({elapsed:.1f}s elapsed).")

    print(f"Collection complete: {len(new_clean_lines):,} new script-pure sentences obtained.")

    # Append to data/raw/konkani/news.txt
    print(f"Appending {len(new_clean_lines):,} lines to {news_file}...")
    with open(news_file, "a", encoding="utf-8") as f:
        for l in new_clean_lines:
            f.write(l + "\n")

    # Update processed sentences in data/processed/konkani/sentences.txt
    print(f"Segmenting and adding new sentences to {processed_file}...")
    with open(processed_file, "a", encoding="utf-8") as pf:
        for raw_l in new_clean_lines:
            norm_l = normalize_text(raw_l)
            sents = segment_sentences(norm_l, "konkani", ["डॉ.", "श्री."])
            for s in sents:
                tokens = s.split()
                if len(tokens) >= 3:
                    pf.write(" ".join(tokens) + "\n")

    # Re-run corpus statistics report
    print("Re-running corpus stats to update reports/corpus_stats.md...")
    stats = run_corpus_stats()
    print("Corpus stats updated successfully.")

    return {
        "new_lines_collected": len(new_clean_lines),
        "total_konkani_sentences": stats.get("konkani_total_sentences", 0),
        "floor_met": stats.get("konkani_total_sentences", 0) >= PHASE1_FLOOR,
    }


if __name__ == "__main__":
    expand_konkani_corpus()
