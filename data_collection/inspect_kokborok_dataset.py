"""
Phase 9: Kokborok Dataset Ingestion & Inspection Script.

Performs complete, assumption-free structural, statistical, and quality inspection
of all files in the data/raw/kokborok directory.

Guarantees:
- Strictly read-only against data/raw/kokborok.
- Pre- and post-inspection SHA-256 verification.
- Comprehensive extraction of format, schema, script, vocabulary, lengths, and duplicates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import openpyxl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "kokborok"


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def inspect_all_files() -> dict[str, Any]:
    print("=" * 80)
    print("PHASE 9: KOKBOROK DATASET INGESTION & INSPECTION")
    print("=" * 80)

    # 1. Pre-inspection SHA-256 lock
    print("\n[Step 1] Computing pre-inspection SHA-256 checksums...")
    pre_hashes = {}
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_file():
            h = compute_file_sha256(p)
            pre_hashes[p.name] = h
            print(f"  {p.name:35s} | {p.stat().st_size:8d} bytes | SHA-256: {h[:16]}...")

    files_profile = {}
    all_kokborok_sentences = []
    file_sentence_map = {}

    # 2. Detailed inspection of each file
    print("\n[Step 2] Inspecting sheets, schemas, and contents...")
    for p in sorted(DATA_DIR.iterdir()):
        if not p.is_file():
            continue

        wb = openpyxl.load_workbook(p, data_only=True)
        file_info = {
            "file_name": p.name,
            "file_size_bytes": p.stat().st_size,
            "sha256": pre_hashes[p.name],
            "sheets": {},
        }

        file_kokborok_sents = []

        for sname in wb.sheetnames:
            sheet = wb[sname]
            is_worksheet = isinstance(sheet, openpyxl.worksheet.worksheet.Worksheet)
            if not is_worksheet:
                file_info["sheets"][sname] = {
                    "sheet_type": type(sheet).__name__,
                    "is_worksheet": False,
                    "description": "Non-worksheet object (e.g. Chartsheet)",
                }
                continue

            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                file_info["sheets"][sname] = {
                    "sheet_type": "Worksheet",
                    "is_worksheet": True,
                    "total_rows": 0,
                    "empty": True,
                }
                continue

            headers = [str(c).strip() if c is not None else f"Col_{i}" for i, c in enumerate(rows[0])]
            data_rows = rows[1:]

            # Column non-empty counts
            col_counts = {}
            for c_idx, h in enumerate(headers):
                non_empty = sum(
                    1
                    for r in data_rows
                    if len(r) > c_idx and r[c_idx] is not None and str(r[c_idx]).strip() != ""
                )
                col_counts[h] = non_empty

            file_info["sheets"][sname] = {
                "sheet_type": "Worksheet",
                "is_worksheet": True,
                "total_rows_including_header": len(rows),
                "data_rows_count": len(data_rows),
                "headers": headers,
                "column_non_empty_counts": col_counts,
                "sample_rows": [list(r) for r in data_rows[:3]],
            }

            # Extract Kokborok sentences
            if p.name == "English-Kokborok 3rd Dataset.xlsx":
                # Col 1: Kokborok Sentences
                for r in data_rows:
                    if len(r) > 1 and r[1] is not None and str(r[1]).strip():
                        s = str(r[1]).strip()
                        file_kokborok_sents.append(s)
                        all_kokborok_sentences.append((s, p.name, sname, "parallel_translation"))

            elif p.name == "Kokborok second dataset (1).xlsx":
                # Col 1: Kokborok Sentences
                for r in data_rows:
                    if len(r) > 1 and r[1] is not None and str(r[1]).strip():
                        s = str(r[1]).strip()
                        file_kokborok_sents.append(s)
                        all_kokborok_sentences.append((s, p.name, sname, "parallel_translation"))

            elif p.name == "Kokborok Sentiment data.xlsx":
                # Col 0: Kokborok Sentences
                for r in data_rows:
                    if len(r) > 0 and r[0] is not None and str(r[0]).strip():
                        s = str(r[0]).strip()
                        file_kokborok_sents.append(s)
                        all_kokborok_sentences.append((s, p.name, sname, "sentiment_sarcasm"))

            elif p.name == "Kokborok Translations (1).xlsx":
                # Col 1: KOKBOROK TRANSLATIONS
                for r in data_rows:
                    if len(r) > 1 and r[1] is not None and str(r[1]).strip():
                        s = str(r[1]).strip()
                        file_kokborok_sents.append(s)
                        all_kokborok_sentences.append((s, p.name, sname, "parallel_translation"))

            elif p.name == "Kokborok-dataset.xlsx":
                # Headerless or row 0 is data! In this file, all rows including row 0 are code-switched lines
                for r in rows:
                    if len(r) > 0 and r[0] is not None and str(r[0]).strip():
                        s = str(r[0]).strip()
                        file_kokborok_sents.append(s)
                        all_kokborok_sentences.append((s, p.name, sname, "code_switched_lid"))

            elif p.name == "Question Answering.xlsx":
                # Check if col 1 has any data
                for r in data_rows:
                    if len(r) > 1 and r[1] is not None and str(r[1]).strip():
                        s = str(r[1]).strip()
                        file_kokborok_sents.append(s)
                        all_kokborok_sentences.append((s, p.name, sname, "qa_answer"))

        file_sentence_map[p.name] = file_kokborok_sents
        files_profile[p.name] = file_info

    # 3. Unicode Character and Script Analysis
    print("\n[Step 3] Analyzing character encoding and Unicode scripts...")
    script_counts = Counter()
    char_code_counts = Counter()
    bengali_codepoints = []
    mojibake_chars = []

    for s, fname, sname, category in all_kokborok_sentences:
        for ch in s:
            cp = ord(ch)
            script_name = unicodedata.name(ch, "UNKNOWN")
            char_code_counts[ch] += 1

            if 0x0980 <= cp <= 0x09FF:
                script_counts["Bengali"] += 1
                bengali_codepoints.append((ch, hex(cp), script_name))
            elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
                script_counts["Latin_Alphabet"] += 1
            elif cp in (0x0020, 0x0009, 0x000A, 0x000D):
                script_counts["Whitespace"] += 1
            elif cp < 128:
                script_counts["ASCII_Punctuation_or_Symbol"] += 1
            elif cp in (0x2018, 0x2019, 0x201C, 0x201D):
                script_counts["Unicode_Punctuation (Quotes)"] += 1
            elif 0x80 <= cp <= 0xFF:
                script_counts["Latin-1 Supplement (Mojibake candidates)"] += 1
                mojibake_chars.append((ch, hex(cp), script_name))
            else:
                script_counts[f"Other ({script_name[:20]})"] += 1

    print(f"  Latin Letters: {script_counts['Latin_Alphabet']:,d}")
    print(f"  ASCII Punctuation/Symbols: {script_counts['ASCII_Punctuation_or_Symbol']:,d}")
    print(f"  Bengali Script Characters: {script_counts['Bengali']:,d}")
    print(f"  Unicode Quotes: {script_counts['Unicode_Punctuation (Quotes)']:,d}")
    print(f"  Mojibake / Latin-1 Extended: {script_counts['Latin-1 Supplement (Mojibake candidates)']:,d}")

    # 4. Statistical Analysis of Kokborok Sentences
    print("\n[Step 4] Computing statistical distributions and vocabulary...")
    # Word tokenization (standard whitespace & punctuation regex)
    word_pattern = re.compile(r"[a-zA-Z0-9_\-\']+|[^\s\w]")

    clean_sents = []
    tokenized_sents = []
    vocab_counter = Counter()
    vocab_lower_counter = Counter()
    lengths = []

    # Tag analysis for code-switched file
    lid_tags = Counter()

    for s, fname, sname, category in all_kokborok_sentences:
        clean_s = s.strip()
        if not clean_s:
            continue
        clean_sents.append(clean_s)

        # Tokenize
        if category == "code_switched_lid":
            # Extract raw tokens and language tags
            toks = clean_s.split()
            sent_words = []
            for t in toks:
                if "/" in t:
                    parts = t.rsplit("/", 1)
                    w, tag = parts[0], parts[1]
                    lid_tags[tag] += 1
                    sent_words.append(w)
                else:
                    sent_words.append(t)
            tokens = sent_words
        else:
            tokens = word_pattern.findall(clean_s)

        tokenized_sents.append(tokens)
        lengths.append(len(tokens))
        for tok in tokens:
            vocab_counter[tok] += 1
            vocab_lower_counter[tok.lower()] += 1

    lengths.sort()
    n_sents = len(lengths)
    total_tokens = sum(lengths)
    mean_len = total_tokens / n_sents if n_sents > 0 else 0
    median_len = lengths[n_sents // 2] if n_sents > 0 else 0
    min_len = lengths[0] if n_sents > 0 else 0
    max_len = lengths[-1] if n_sents > 0 else 0
    q1_len = lengths[int(n_sents * 0.25)] if n_sents > 0 else 0
    q3_len = lengths[int(n_sents * 0.75)] if n_sents > 0 else 0
    var_len = sum((l - mean_len) ** 2 for l in lengths) / n_sents if n_sents > 0 else 0
    std_len = math.sqrt(var_len)

    # 5. Duplication Analysis
    print("\n[Step 5] Analyzing duplicates and near-duplicates...")
    exact_counter = Counter(clean_sents)
    exact_duplicates = sum(count - 1 for count in exact_counter.values() if count > 1)
    unique_exact_sents = len(exact_counter)

    # Normalized near-duplicate (lowercased, stripped punctuation, normalized spaces)
    norm_sents = [re.sub(r"[^\w\s]", "", s.lower()).strip() for s in clean_sents]
    norm_counter = Counter(norm_sents)
    near_duplicates = sum(count - 1 for count in norm_counter.values() if count > 1)
    unique_norm_sents = len(norm_counter)

    # 6. Suffix and Morphological Pattern Cross-Check
    print("\n[Step 6] Cross-checking Kokborok morphological case markers...")
    case_suffixes = {
        "-no (Acc/Dat)": re.compile(r"^[a-zA-Z]{3,}no$", re.IGNORECASE),
        "-ni (Gen)": re.compile(r"^[a-zA-Z]{3,}ni$", re.IGNORECASE),
        "-o (Loc)": re.compile(r"^[a-zA-Z]{3,}o$", re.IGNORECASE),
        "-bai (Inst)": re.compile(r"^[a-zA-Z]{3,}bai$", re.IGNORECASE),
        "-rok (Plural)": re.compile(r"^[a-zA-Z]{3,}rok$", re.IGNORECASE),
        "-song (Plural Human)": re.compile(r"^[a-zA-Z]{3,}song$", re.IGNORECASE),
        "-bagwi (Benefactive/Postposition)": re.compile(r"^[a-zA-Z]{3,}bagwi$", re.IGNORECASE),
    }

    suffix_counts = Counter()
    for tok, cnt in vocab_counter.items():
        for s_name, regex in case_suffixes.items():
            if regex.match(tok):
                suffix_counts[s_name] += cnt

    # 7. Post-inspection SHA-256 verification (Read-only proof)
    print("\n[Step 7] Re-verifying SHA-256 checksums to confirm read-only integrity...")
    post_hashes = {}
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_file():
            h = compute_file_sha256(p)
            post_hashes[p.name] = h
            if h != pre_hashes[p.name]:
                raise RuntimeError(f"CRITICAL INTEGRITY VIOLATION! File {p.name} was modified!")
            print(f"  [PASS] {p.name:35s} SHA-256 UNMODIFIED")

    summary_stats = {
        "total_files": len(pre_hashes),
        "total_kokborok_sentences": n_sents,
        "unique_exact_sentences": unique_exact_sents,
        "exact_duplicates": exact_duplicates,
        "unique_normalized_sentences": unique_norm_sents,
        "near_duplicates": near_duplicates,
        "total_tokens": total_tokens,
        "vocabulary_case_sensitive": len(vocab_counter),
        "vocabulary_case_insensitive": len(vocab_lower_counter),
        "sentence_length_distribution": {
            "min": min_len,
            "max": max_len,
            "mean": round(mean_len, 2),
            "median": median_len,
            "std_dev": round(std_len, 2),
            "q1": q1_len,
            "q3": q3_len,
        },
        "script_distribution": dict(script_counts),
        "morphological_suffix_matches": dict(suffix_counts),
        "code_switched_lid_tags": dict(lid_tags),
        "files_profile": files_profile,
    }

    # Save summary json to scratch for report generation
    out_json = PROJECT_ROOT / "scratch" / "kokborok_inspection_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    print(f"\nInspection statistics saved to: {out_json.relative_to(PROJECT_ROOT)}")

    return summary_stats


if __name__ == "__main__":
    inspect_all_files()
