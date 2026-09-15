"""
News and press text collection script for Konkani, Assamese, and Sinhala.

Collects text from publicly accessible, ToS-respecting news sources,
public RSS feeds, and open press releases without pretrained models.
"""

from __future__ import annotations

import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import BaseConfig, load_config, parse_cli_args
from utils.logger import ExperimentLogger
from utils.seed import set_global_seed


class HTMLTextExtractor(HTMLParser):
    """Strip HTML markup, script tags, style tags, and extract clean text."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []
        self.ignore_tag = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav", "header", "footer", "aside"}:
            self.ignore_tag = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "header", "footer", "aside"}:
            self.ignore_tag = False
        elif tag in {"p", "br", "div", "li", "h1", "h2", "h3", "h4"}:
            self.texts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignore_tag:
            self.texts.append(data)

    def get_text(self) -> str:
        raw = "".join(self.texts)
        # Normalize whitespace
        lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def clean_html(raw_html: str) -> str:
    """Helper to clean raw HTML string."""
    parser = HTMLTextExtractor()
    parser.feed(raw_html)
    return parser.get_text()


NEWS_FEEDS: dict[str, list[dict[str, str]]] = {
    "sinhala": [
        {"name": "Ada Derana Sinhala", "url": "http://sinhala.adaderana.lk/rss.php", "type": "rss"},
        {"name": "BBC Sinhala", "url": "https://feeds.bbci.co.uk/sinhala/rss.xml", "type": "rss"},
        {"name": "Dinamina", "url": "http://www.dinamina.lk/rss.xml", "type": "rss"},
    ],
    "assamese": [
        {"name": "Niyomiya Barta", "url": "https://niyomiyabarta.com/feed/", "type": "rss"},
        {"name": "Dainik Janambhumi", "url": "https://dainikjanambhumi.co.in/feed/", "type": "rss"},
    ],
    "konkani": [
        {"name": "Bhaangarbhuin News Portal", "url": "https://bhaangarbhuin.com/feed/", "type": "rss"},
    ],
}


def fetch_rss_feed(feed_info: dict[str, str]) -> list[str]:
    """Fetch items from an RSS feed and extract title + description text."""
    url = feed_info["url"]
    name = feed_info["name"]
    items_text: list[str] = []

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NLPResearchBot/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            content = resp.read()
        root = ET.fromstring(content)
        for item in root.findall(".//item"):
            title_el = item.find("title")
            desc_el = item.find("description")
            content_el = item.find("{http://purl.org/rss/1.0/modules/content/}encoded")

            title = clean_html(title_el.text) if title_el is not None and title_el.text else ""
            desc = clean_html(desc_el.text) if desc_el is not None and desc_el.text else ""
            body = clean_html(content_el.text) if content_el is not None and content_el.text else ""

            for text_chunk in [title, desc, body]:
                if text_chunk:
                    for line in text_chunk.splitlines():
                        line = line.strip()
                        if line:
                            items_text.append(line)
        print(f"[{name}] Fetched {len(items_text)} news text lines from RSS.")
    except Exception as exc:
        print(f"[{name}] RSS fetch warning ({url}): {exc}")

    return items_text


def collect_news_for_language(lang: str, output_dir: Path) -> int:
    """Collect all available news text for a given language into news.txt."""
    output_path = output_dir / lang / "news.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    collected_lines: list[str] = []

    # 1. Fetch live RSS feeds
    feeds = NEWS_FEEDS.get(lang, [])
    for feed in feeds:
        lines = fetch_rss_feed(feed)
        collected_lines.extend(lines)

    # 2. Add vetted open press articles and news archives for volume stability
    # If live feeds yielded few lines due to transient network rate limits, augment with curated open news bulletins
    if len(collected_lines) < 500:
        # Fetch public news archive texts or open regional bulletins
        open_archive_urls = {
            "konkani": [
                "https://raw.githubusercontent.com/indicnlp/indicnlp-corpus/master/konkani/news_sample.txt",
            ],
            "assamese": [
                "https://raw.githubusercontent.com/indicnlp/indicnlp-corpus/master/assamese/news_sample.txt",
            ],
            "sinhala": [
                "https://raw.githubusercontent.com/indicnlp/indicnlp-corpus/master/sinhala/news_sample.txt",
            ],
        }
        for arch_url in open_archive_urls.get(lang, []):
            try:
                req = urllib.request.Request(arch_url, headers={"User-Agent": "NLPResearchBot/1.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    for line in resp.read().decode("utf-8", errors="ignore").splitlines():
                        if line.strip():
                            collected_lines.append(line.strip())
            except Exception:
                pass

    with open(output_path, "w", encoding="utf-8") as f:
        for line in collected_lines:
            f.write(line + "\n")

    print(f"[{lang.upper()}] Wrote {len(collected_lines)} lines to {output_path}")
    return len(collected_lines)


def run_news_scraping(config_path: str = "configs/base_config.yaml", overrides: list[str] | None = None) -> dict[str, Any]:
    cfg: BaseConfig = load_config(config_path, overrides=overrides)
    set_global_seed(cfg.seed)

    logger = ExperimentLogger(log_dir=cfg.logging.log_dir, backend=cfg.logging.backend)
    raw_dir = PROJECT_ROOT / "data" / "raw"

    stats: dict[str, Any] = {}
    for lang in cfg.project.languages:
        lines_count = collect_news_for_language(lang, raw_dir)
        stats[f"{lang}_news_lines"] = lines_count

    logger.log_run(
        config_path=config_path,
        metrics=stats,
        dataset_version="v0.1-news-raw",
    )
    return stats


if __name__ == "__main__":
    cfg_file, cli_overrides = parse_cli_args(sys.argv[1:])
    run_news_scraping(cfg_file, cli_overrides)
