"""
Public Kokborok Corpus Ingestion Pipeline.

Fetches open-access raw text from:
1. Hugging Face: sdmy/kokborok (smoldoc__en_trp.json, smolsent__en_trp.json)
2. Wikimedia Incubator: Kokborok Wikipedia test articles (Wp/trp)

Applies canonical Kokborok normalization (NFC, Unicode control strip, quote repair).
Strictly compliant with project zero-pretrained-model constraint.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.kokborok_normalize import normalize_kokborok_text

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "kokborok"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "public_online_sentences.txt"

HF_BASE = "https://huggingface.co/datasets/sdmy/kokborok/raw/main"
INCUBATOR_API = "https://incubator.wikimedia.org/w/api.php"


def fetch_huggingface_sentences() -> list[str]:
    print("[1/2] Fetching Kokborok corpus from Hugging Face (sdmy/kokborok)...")
    sentences: list[str] = []

    # 1. smoldoc
    for fname in ["smoldoc__en_trp.json", "smolsent__en_trp.json"]:
        url = f"{HF_BASE}/{fname}"
        req = urllib.request.Request(url, headers={"User-Agent": "KokborokResearchBot/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                count = 0
                for item in data:
                    if "trgs" in item and isinstance(item["trgs"], list):
                        for s in item["trgs"]:
                            if s and isinstance(s, str):
                                clean = normalize_kokborok_text(s.strip())
                                if len(clean) > 8 and " " in clean:
                                    sentences.append(clean)
                                    count += 1
                    elif "src" in item and item.get("tl") == "trp":
                        s = item.get("src", "")
                        clean = normalize_kokborok_text(s.strip())
                        if len(clean) > 8 and " " in clean:
                            sentences.append(clean)
                            count += 1
                print(f"  - Downloaded {fname}: extracted {count} sentences.")
        except Exception as e:
            print(f"  - Error downloading {fname}: {e}")

    return sentences


def fetch_wikimedia_incubator_sentences() -> list[str]:
    print("[2/2] Fetching Kokborok Wikipedia articles from Wikimedia Incubator...")
    sentences: list[str] = []

    list_url = f"{INCUBATOR_API}?action=query&list=allpages&apprefix=Wp/trp/&aplimit=100&format=json"
    req = urllib.request.Request(list_url, headers={"User-Agent": "KokborokResearchBot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("allpages", [])
            print(f"  - Found {len(pages)} Wikimedia Incubator pages.")
    except Exception as e:
        print(f"  - Error listing Incubator pages: {e}")
        return sentences

    page_ids = [str(p["pageid"]) for p in pages]
    # Fetch page contents in batches
    for i in range(0, len(page_ids), 20):
        batch = "|".join(page_ids[i : i + 20])
        content_url = f"{INCUBATOR_API}?action=query&prop=extracts&explaintext=1&pageids={batch}&format=json"
        req = urllib.request.Request(content_url, headers={"User-Agent": "KokborokResearchBot/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                cdata = json.loads(resp.read().decode("utf-8"))
                for pid, pinfo in cdata.get("query", {}).get("pages", {}).items():
                    extract = pinfo.get("extract", "")
                    for line in extract.splitlines():
                        line = normalize_kokborok_text(line.strip())
                        if len(line) > 15 and " " in line:
                            # Split into sentences on standard terminals
                            for s in re.split(r"(?<=[.!?।])\s+", line):
                                s = s.strip()
                                if len(s) > 10 and " " in s:
                                    sentences.append(s)
        except Exception as e:
            print(f"  - Error fetching content batch: {e}")

    print(f"  - Extracted {len(sentences)} sentences from Wikimedia Incubator.")
    return sentences


def main() -> None:
    all_sents = []
    all_sents.extend(fetch_huggingface_sentences())
    all_sents.extend(fetch_wikimedia_incubator_sentences())

    # Deduplicate while preserving order
    seen = set()
    unique_sents = []
    for s in all_sents:
        if s not in seen:
            seen.add(s)
            unique_sents.append(s)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for s in unique_sents:
            f.write(s + "\n")

    print(f"\n[SUCCESS] Saved {len(unique_sents):,} unique public Kokborok sentences to:")
    print(f"  {OUT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
