"""
Silver Dataset Generator for Kokborok NER.

Labels raw Kokborok sentences from:
1. Public online datasets (sdmy/kokborok, Wikimedia Incubator): data/raw/kokborok/public_online_sentences.txt
2. Local processed sentences: data/processed/kokborok/sentences.txt

Applies:
- High-precision gazetteer dictionary lookup (PER, LOC, ORG)
- Suffix-invariant stem matching for Kokborok case markers (-o, -ni, -no, -bai, -rok)
- Cultural events & facilities mapped to CoNLL-2003 MISC / LOC
- BIO transition validation and punctuation masking

Generates:
- data/silver/kokborok/train.conll
- data/processed/kokborok_augmented_silver_train.conll (Gold Train + Silver Train)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.kokborok_tokenizer import KokborokTokenizer
from weak_supervision.gazetteer_matcher import PUNCTUATION_TOKENS

GAZ_DIR = PROJECT_ROOT / "data" / "gazetteers" / "kokborok"
RAW_PUBLIC_FILE = PROJECT_ROOT / "data" / "raw" / "kokborok" / "public_online_sentences.txt"
LOCAL_SENT_FILE = PROJECT_ROOT / "data" / "processed" / "kokborok" / "sentences.txt"
GOLD_TRAIN_FILE = PROJECT_ROOT / "data" / "gold" / "kokborok" / "train.conll"

SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "kokborok"
SILVER_DIR.mkdir(parents=True, exist_ok=True)
SILVER_CONLL = SILVER_DIR / "train.conll"
COMBINED_CONLL = PROJECT_ROOT / "data" / "processed" / "kokborok_augmented_silver_train.conll"

KOKBOROK_SUFFIXES = ("-ni", "-o", "-no", "-bai", "-rok", "-song", "-rog", "ni", "no")

# Extra events and facilities for CoNLL-2003 mapping
EXTRA_EVENTS = [
    ["Buisu"], ["Kharchi", "Puja"], ["Kokborok", "Day"], ["Garia", "Puja"],
    ["Ker", "Puja"], ["Wangala"], ["Hojagiri"], ["Durga", "Puja"],
    ["Diwali"], ["Christmas"], ["Id"]
]
EXTRA_FACILITIES = [
    ["Ujjayanta", "Palace"], ["Neermahal"],
    ["Agartala", "Airport"], ["MBB", "Airport"],
    ["Tripura", "University"], ["MBB", "University"]
]


def load_gazetteers() -> dict[str, list[list[str]]]:
    entities: dict[str, list[list[str]]] = {"ORG": [], "PER": [], "LOC": [], "MISC": []}

    mapping = {"orgs.txt": "ORG", "persons.txt": "PER", "places.txt": "LOC"}
    for fname, etype in mapping.items():
        fpath = GAZ_DIR / fname
        if fpath.is_file():
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    entry = line.strip()
                    if entry and len(entry) >= 2:
                        toks = entry.split()
                        entities[etype].append(toks)

    # Extra events -> MISC
    for ev in EXTRA_EVENTS:
        entities["MISC"].append(ev)

    # Extra facilities -> LOC
    for fac in EXTRA_FACILITIES:
        entities["LOC"].append(fac)

    # Sort descending by length for longest match first
    for etype in entities:
        entities[etype].sort(key=len, reverse=True)

    return entities


def strip_suffix(tok: str) -> str:
    for suf in KOKBOROK_SUFFIXES:
        if tok.endswith(suf) and len(tok) > len(suf) + 2:
            return tok[:-len(suf)].rstrip("-")
    return tok


def label_tokens(tokens: list[str], entities: dict[str, list[list[str]]]) -> list[str]:
    n = len(tokens)
    labels = ["O"] * n
    matched = [False] * n

    # Also build stripped tokens for suffix-invariant matching
    stem_tokens = [strip_suffix(t) for t in tokens]

    for etype in ["ORG", "MISC", "PER", "LOC"]:
        for span in entities[etype]:
            slen = len(span)
            if slen > n:
                continue

            for i in range(n - slen + 1):
                if any(matched[i + k] for k in range(slen)):
                    continue

                # 1. Exact match on raw tokens
                raw_match = (tokens[i : i + slen] == span)

                # 2. Suffix-invariant match (last token has suffix)
                suffix_match = False
                if slen == 1:
                    suffix_match = (stem_tokens[i].lower() == span[0].lower())
                else:
                    if (tokens[i : i + slen - 1] == span[:-1] and
                        stem_tokens[i + slen - 1].lower() == span[-1].lower()):
                        suffix_match = True

                if raw_match or suffix_match:
                    if tokens[i] in PUNCTUATION_TOKENS or tokens[i + slen - 1] in PUNCTUATION_TOKENS:
                        continue

                    labels[i] = f"B-{etype}"
                    matched[i] = True
                    for k in range(1, slen):
                        labels[i + k] = f"I-{etype}"
                        matched[i + k] = True

    # Punctuation invariant
    for i, t in enumerate(tokens):
        if t in PUNCTUATION_TOKENS:
            labels[i] = "O"

    return labels


def main() -> None:
    tokenizer = KokborokTokenizer()
    entities = load_gazetteers()

    total_gaz = sum(len(v) for v in entities.values())
    print(f"Loaded {total_gaz} gazetteer patterns across ORG, PER, LOC, MISC.")

    # Collect sentences
    all_sentences: list[str] = []
    if RAW_PUBLIC_FILE.is_file():
        with open(RAW_PUBLIC_FILE, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    all_sentences.append(s)

    if LOCAL_SENT_FILE.is_file():
        with open(LOCAL_SENT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    all_sentences.append(s)

    print(f"Total raw sentences collected: {len(all_sentences):,}")

    silver_samples: list[list[tuple[str, str]]] = []
    entity_counts = {"PER": 0, "LOC": 0, "ORG": 0, "MISC": 0}

    for s in all_sentences:
        tokens = tokenizer.tokenize_words(s)
        if not tokens or len(tokens) < 3 or len(tokens) > 60:
            continue

        labels = label_tokens(tokens, entities)
        has_entity = any(l != "O" for l in labels)

        # Count entities
        for l in labels:
            if l.startswith("B-"):
                etype = l[2:]
                if etype in entity_counts:
                    entity_counts[etype] += 1

        # Keep all sentences with entities, plus 10% background O sentences
        if has_entity or (len(silver_samples) % 10 == 0):
            sample = list(zip(tokens, labels))
            silver_samples.append(sample)

    print(f"\nWeak Supervision Complete:")
    print(f"  Total Silver Sentences Generated: {len(silver_samples):,}")
    print(f"  Entity counts: {entity_counts}")

    # Write data/silver/kokborok/train.conll
    with open(SILVER_CONLL, "w", encoding="utf-8") as f:
        for sent in silver_samples:
            for tok, tag in sent:
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")
    print(f"[SAVED] Silver dataset: {SILVER_CONLL.relative_to(PROJECT_ROOT)}")

    # Merge Gold Train + Silver Train for complete training dataset
    gold_lines = []
    if GOLD_TRAIN_FILE.is_file():
        with open(GOLD_TRAIN_FILE, "r", encoding="utf-8") as f:
            gold_lines = f.readlines()

    with open(COMBINED_CONLL, "w", encoding="utf-8") as f:
        # Write Gold first (repeated 3x for higher weighting)
        for _ in range(3):
            for line in gold_lines:
                f.write(line)
            if gold_lines and not gold_lines[-1].endswith("\n\n"):
                f.write("\n")
        # Append Silver
        for sent in silver_samples:
            for tok, tag in sent:
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")

    print(f"[SAVED] Combined Training dataset: {COMBINED_CONLL.relative_to(PROJECT_ROOT)} ({COMBINED_CONLL.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
