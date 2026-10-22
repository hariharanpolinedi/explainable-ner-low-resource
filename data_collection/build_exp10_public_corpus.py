"""
Experiment 10: Kokborok Public Corpus Expansion, Cleaning, Provenance Tracking & NER Candidate Discovery.
Harvests text from all approved public sources, performs deduplication, audits quality,
computes vocabulary growth over Gold-Train, extracts candidate entity sentences, and generates all required artifacts.
"""

import csv
import hashlib
import io
import json
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path("c:/Users/harih/Desktop/NLP-MARK-3")
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

RAW_CORPUS_DIR = PROJECT_ROOT / "data" / "raw" / "kokborok_public_corpus"
RAW_CORPUS_DIR.mkdir(parents=True, exist_ok=True)

PROC_DIR = PROJECT_ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "kokborok"

# 1. Load Gold Sets to compute baseline vocab and verify overlap
def load_conll(path):
    sentences = []
    curr_toks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if curr_toks:
                    sentences.append(curr_toks)
                    curr_toks = []
            else:
                parts = line.split()
                if len(parts) >= 1:
                    curr_toks.append(parts[0])
        if curr_toks:
            sentences.append(curr_toks)
    return sentences

train_gold_sents = load_conll(GOLD_DIR / "train.conll")
dev_gold_sents = load_conll(GOLD_DIR / "dev.conll")

gold_train_vocab = set(t.lower() for s in train_gold_sents for t in s)
gold_train_sent_texts = set(" ".join(s).strip() for s in train_gold_sents)
gold_dev_sent_texts = set(" ".join(s).strip() for s in dev_gold_sents)

print(f"Gold Train Vocab: {len(gold_train_vocab)} unique tokens across {len(train_gold_sents)} sentences.")

# Helper for text hash
def compute_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# Helper for minimal Kokborok normalization
def minimal_normalize(text):
    text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\t', ' ')
    # Remove code-mixing annotations if present
    text = re.sub(r'/(?:trp|en|bn|hi)\b', '', text)
    # Normalize multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------------------------------------
# STEP 1: HARVESTING FROM APPROVED PUBLIC SOURCES
# -------------------------------------------------------------
raw_records = []

# SOURCE 1: SpeeD-TB (speed-tb/kokborok) - Gated metadata recording
source_inventory = [
    {
        "source_id": "SRC_1_SPEED_TB",
        "dataset_name": "SpeeD-TB Kokborok Speech Corpus",
        "url": "https://huggingface.co/datasets/speed-tb/kokborok",
        "license": "CC-BY-NC-SA 4.0",
        "permitted_use": "Non-commercial academic research with attribution and share-alike",
        "access_status": "Gated / Interactive HF agreement required",
        "rows_available": "200+ audio hours (~15,000 utterances)",
        "domains": "education, agriculture, healthcare, oral history, sports, science/technology",
        "text_columns": "transcription (Roman/IPA)",
        "status": "Documented in inventory (requires gated auth for bulk download)"
    },
    {
        "source_id": "SRC_2_SDMY",
        "dataset_name": "Kokborok Digitalisation (sdmy/kokborok)",
        "url": "https://huggingface.co/datasets/sdmy/kokborok",
        "license": "MIT License",
        "permitted_use": "Commercial and non-commercial research with attribution",
        "access_status": "Public / Open Access",
        "rows_available": "457 docs + 57 sents + 3,995 phrases",
        "domains": "General domain, parallel translations, local dialect corrections",
        "text_columns": "trgs, src, correction",
        "status": "Harvested"
    },
    {
        "source_id": "SRC_3_QA",
        "dataset_name": "Kokborok QA Corpus (agnivamaiti/kokborok-qa)",
        "url": "https://huggingface.co/datasets/agnivamaiti/kokborok-qa",
        "license": "CC-BY-NC 4.0",
        "permitted_use": "Non-commercial research use with attribution",
        "access_status": "Public / Open Access",
        "rows_available": "4,943 QA pairs (~9,886 sentences)",
        "domains": "Conversational, administrative, cultural, general QA",
        "text_columns": "prompt, completion",
        "status": "Harvested"
    },
    {
        "source_id": "SRC_4_KOKLAI",
        "dataset_name": "KokLai Portal (koklai.in)",
        "url": "https://koklai.in",
        "license": "LICENSE_UNCLEAR (Proprietary educational web portal)",
        "permitted_use": "Web interaction only; no open downloadable API or bulk dump",
        "access_status": "Web UI (No open downloadable data dump)",
        "rows_available": "N/A",
        "domains": "Web dictionary & educational tools",
        "text_columns": "N/A",
        "status": "Audited (No downloadable bulk dump available)"
    },
    {
        "source_id": "SRC_5_NE_ASR",
        "dataset_name": "Northeast ASR/TTS Kokborok (sulabhkatiyar/ne-asr-dataset-trp)",
        "url": "https://huggingface.co/datasets/sulabhkatiyar/ne-asr-dataset-trp",
        "license": "CC-BY 4.0",
        "permitted_use": "Commercial and non-commercial research use with attribution",
        "access_status": "Public / Open Access",
        "rows_available": "1,000+ speech transcriptions",
        "domains": "Spoken Kokborok narration, field dialogues",
        "text_columns": "text",
        "status": "Harvested"
    },
    {
        "source_id": "SRC_6_WIKIMEDIA",
        "dataset_name": "Wikimedia Incubator Kokborok Wikipedia (Wp/trp)",
        "url": "https://incubator.wikimedia.org/wiki/Wp/trp/",
        "license": "CC-BY-SA 3.0 / 4.0",
        "permitted_use": "Open educational and research use with attribution and share-alike",
        "access_status": "Public MediaWiki API",
        "rows_available": "26 encyclopedia test articles (~321 sentences)",
        "domains": "History, science, culture, geography, language",
        "text_columns": "extract",
        "status": "Harvested"
    },
    {
        "source_id": "SRC_7_LOCAL_SPREADSHEETS",
        "dataset_name": "Local Kokborok Academic Spreadsheets (Kokborok_datasets/*.xlsx)",
        "url": "Local Academic Repository",
        "license": "Academic Research License",
        "permitted_use": "Academic research use within project",
        "access_status": "Local files",
        "rows_available": "6 spreadsheets (~4,019 sentences)",
        "domains": "Translations, code-mixing, sentiment, linguistic QA",
        "text_columns": "Kokborok Sentences, Translations, Answers",
        "status": "Harvested"
    }
]

print("\n--- Harvesting Data from Approved Public Sources ---")

# 1. Harvest sdmy/kokborok
# smoldoc
url_smoldoc = "https://huggingface.co/datasets/sdmy/kokborok/raw/main/smoldoc__en_trp.json"
try:
    req = urllib.request.Request(url_smoldoc, headers={"User-Agent": "KokborokResearchBot/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        docs = json.loads(resp.read().decode("utf-8"))
        # Save raw copy
        with open(RAW_CORPUS_DIR / "sdmy_smoldoc.json", "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False)
        for doc in docs:
            doc_id = doc.get("id", "")
            trgs = doc.get("trgs", [])
            for i, s in enumerate(trgs):
                if s and isinstance(s, str):
                    raw_records.append({
                        "source": "sdmy/kokborok (smoldoc)",
                        "source_file": "smoldoc__en_trp.json",
                        "source_row_id": f"{doc_id}_s{i}",
                        "license": "MIT",
                        "text": s
                    })
    print(f"Harvested sdmy smoldoc: {len(raw_records)} records so far.")
except Exception as e:
    print(f"Error harvesting smoldoc: {e}")

# smolsent
url_smolsent = "https://huggingface.co/datasets/sdmy/kokborok/raw/main/smolsent__en_trp.json"
try:
    req = urllib.request.Request(url_smolsent, headers={"User-Agent": "KokborokResearchBot/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        sents = json.loads(resp.read().decode("utf-8"))
        with open(RAW_CORPUS_DIR / "sdmy_smolsent.json", "w", encoding="utf-8") as f:
            json.dump(sents, f, ensure_ascii=False)
        for i, item in enumerate(sents):
            s = item.get("src", "")
            if s and isinstance(s, str):
                raw_records.append({
                    "source": "sdmy/kokborok (smolsent)",
                    "source_file": "smolsent__en_trp.json",
                    "source_row_id": str(item.get("id", i)),
                    "license": "MIT",
                    "text": s
                })
    print(f"Harvested sdmy smolsent: {len(raw_records)} records so far.")
except Exception as e:
    print(f"Error harvesting smolsent: {e}")

# 2. Harvest Kokborok QA (agnivamaiti/kokborok-qa / local qa_train.parquet)
try:
    df_qa = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "kokborok" / "qa_train.parquet")
    df_qa.to_parquet(RAW_CORPUS_DIR / "agnivamaiti_kokborok_qa.parquet")
    for idx, row in df_qa.iterrows():
        p = str(row["prompt"]).replace("User:", "").strip()
        c = str(row["completion"]).replace("Assistant:", "").strip()
        if p:
            raw_records.append({
                "source": "agnivamaiti/kokborok-qa",
                "source_file": "qa_train.parquet (prompt)",
                "source_row_id": f"qa_{idx}_prompt",
                "license": "CC-BY-NC 4.0",
                "text": p
            })
        if c:
            raw_records.append({
                "source": "agnivamaiti/kokborok-qa",
                "source_file": "qa_train.parquet (completion)",
                "source_row_id": f"qa_{idx}_completion",
                "license": "CC-BY-NC 4.0",
                "text": c
            })
    print(f"Harvested Kokborok QA: {len(raw_records)} records so far.")
except Exception as e:
    print(f"Error harvesting QA: {e}")

# 3. Harvest sulabhkatiyar/ne-asr-dataset-trp
try:
    for split_fname in ["validation-00000-of-00001.parquet", "test-00000-of-00001.parquet"]:
        url = f"https://huggingface.co/datasets/sulabhkatiyar/ne-asr-dataset-trp/resolve/main/data/{split_fname}"
        req = urllib.request.Request(url, headers={"User-Agent": "KokborokResearchBot/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            df_asr = pd.read_parquet(io.BytesIO(resp.read()))
            for idx, row in df_asr.iterrows():
                t = str(row.get("text", "")).strip()
                if t:
                    raw_records.append({
                        "source": "sulabhkatiyar/ne-asr-dataset-trp",
                        "source_file": split_fname,
                        "source_row_id": f"asr_{split_fname}_{idx}",
                        "license": "CC-BY 4.0",
                        "text": t
                    })
    print(f"Harvested sulabhkatiyar NE-ASR: {len(raw_records)} records so far.")
except Exception as e:
    print(f"Error harvesting NE-ASR: {e}")

# 4. Harvest Wikimedia Incubator (Wp/trp)
INCUBATOR_API = "https://incubator.wikimedia.org/w/api.php"
try:
    list_url = f"{INCUBATOR_API}?action=query&list=allpages&apprefix=Wp/trp/&aplimit=100&format=json"
    req = urllib.request.Request(list_url, headers={"User-Agent": "KokborokCorpusBot/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        w_data = json.loads(resp.read().decode("utf-8"))
        pages = w_data.get("query", {}).get("allpages", [])
        page_ids = [str(p["pageid"]) for p in pages]
        for i in range(0, len(page_ids), 10):
            batch = "|".join(page_ids[i:i+10])
            curl = f"{INCUBATOR_API}?action=query&prop=extracts&explaintext=1&pageids={batch}&format=json"
            creq = urllib.request.Request(curl, headers={"User-Agent": "KokborokCorpusBot/1.0"})
            with urllib.request.urlopen(creq, timeout=20) as cresp:
                cdat = json.loads(cresp.read().decode("utf-8"))
                for pid, pinfo in cdat.get("query", {}).get("pages", {}).items():
                    title = pinfo.get("title", "")
                    extract = pinfo.get("extract", "")
                    for s_idx, line in enumerate(extract.splitlines()):
                        line = line.strip()
                        if len(line) > 10:
                            for s in re.split(r'(?<=[.!?।])\s+', line):
                                s = s.strip()
                                if len(s) > 10 and " " in s:
                                    raw_records.append({
                                        "source": "Wikimedia Incubator (Wp/trp)",
                                        "source_file": title,
                                        "source_row_id": f"wiki_{pid}_s{s_idx}",
                                        "license": "CC-BY-SA 3.0/4.0",
                                        "text": s
                                    })
    print(f"Harvested Wikimedia Incubator: {len(raw_records)} records so far.")
except Exception as e:
    print(f"Error harvesting Wikimedia: {e}")

# 5. Harvest Local Academic Spreadsheets
import glob
for xls in glob.glob(str(PROJECT_ROOT / "Kokborok_datasets/*.xlsx")):
    fname = Path(xls).name
    try:
        excel = pd.ExcelFile(xls)
        for sname in excel.sheet_names:
            df = pd.read_excel(xls, sheet_name=sname)
            for idx, row in df.iterrows():
                kok_text = ""
                for col in df.columns:
                    clow = str(col).lower()
                    if "kokborok" in clow or "trp" in clow or "translation" in clow or "answer" in clow:
                        kok_text = str(row[col]) if pd.notna(row[col]) else ""
                if not kok_text and len(df.columns) == 1:
                    kok_text = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
                if kok_text and kok_text != "nan":
                    raw_records.append({
                        "source": f"Kokborok_datasets/{fname}",
                        "source_file": f"{fname}:{sname}",
                        "source_row_id": f"xls_{idx}",
                        "license": "Academic Research License",
                        "text": kok_text.strip()
                    })
    except Exception as e:
        print(f"Error harvesting spreadsheet {fname}: {e}")

print(f"\nTotal Raw Records Harvested Across All Sources: {len(raw_records):,}")

# -------------------------------------------------------------
# STEP 2: QUALITY FILTERING & REASON CODING
# -------------------------------------------------------------
processed_records = []
exclusion_counts = Counter()

# Common Kokborok function words for language verification
kok_markers = ['tong', 'wng', 'bai', 'ni', 'no', 'o', 'rok', 'se', 'khe', 'tei', 'hwnwi', 'kaisa', 'bo', 'ang', 'nwng', 'chini', 'bisi', 'sal', 'di', 'khwlai', 'mano', 'swng', 'sak', 'borok', 'khorok', 'chwng', 'norok', 'phaikha', 'tongo', 'rwk']

seen_exact = set()
seen_normalized = set()
duplicates_exact = 0
duplicates_norm = 0

for idx, rec in enumerate(raw_records):
    raw_txt = rec["text"]
    txt_hash = compute_hash(raw_txt)
    clean_txt = minimal_normalize(raw_txt)
    
    tokens = clean_txt.split()
    tok_len = len(tokens)
    
    status = "RETAINED"
    reason = "VALID_KOKBOROK_SENTENCE"
    
    # 1. Empty or whitespace
    if not clean_txt or tok_len == 0:
        status = "EXCLUDED"
        reason = "EMPTY_OR_WHITESPACE"
    # 2. Extremely short (<3 tokens or <10 chars)
    elif tok_len < 3 or len(clean_txt) < 10:
        status = "EXCLUDED"
        reason = "EXTREMELY_SHORT"
    # 3. Extremely long (>150 tokens)
    elif tok_len > 150:
        status = "EXCLUDED"
        reason = "EXTREMELY_LONG"
    # 4. Excessive English / Non-Kokborok
    else:
        has_kok_marker = any(re.search(rf'\b{w}\b', clean_txt, re.IGNORECASE) for w in kok_markers)
        if not has_kok_marker and not any(t.startswith("Kok") or t.startswith("borok") for t in tokens):
            status = "EXCLUDED"
            reason = "NON_KOKBOROK_OR_EXCESSIVE_ENGLISH"

    # Deduplication checks if not already excluded
    if status == "RETAINED":
        if raw_txt in seen_exact:
            status = "EXCLUDED"
            reason = "EXACT_DUPLICATE"
            duplicates_exact += 1
        elif clean_txt in seen_normalized:
            status = "EXCLUDED"
            reason = "NORMALIZED_DUPLICATE"
            duplicates_norm += 1
        else:
            seen_exact.add(raw_txt)
            seen_normalized.add(clean_txt)

    exclusion_counts[reason] += 1
    
    processed_records.append({
        "sentence_id": f"KB_CORP_{idx+1:06d}",
        "source": rec["source"],
        "source_row_id": rec["source_row_id"],
        "source_file": rec["source_file"],
        "license": rec["license"],
        "original_text_hash": txt_hash,
        "cleaned_text": clean_txt,
        "exclusion_status": status,
        "exclusion_reason": reason,
        "token_count": tok_len,
        "tokens": tokens
    })

retained_records = [r for r in processed_records if r["exclusion_status"] == "RETAINED"]
print(f"\n--- Quality Filtering & Deduplication Summary ---")
print(f"Total Raw Inputs:            {len(processed_records):,}")
for reason, cnt in exclusion_counts.most_common():
    print(f"  {reason:<35}: {cnt:6,d}")
print(f"Final Retained Clean Corpus: {len(retained_records):,}")

# -------------------------------------------------------------
# STEP 3: OVERLAP CHECK AGAINST GOLD SETS
# -------------------------------------------------------------
overlap_train = sum(1 for r in retained_records if r["cleaned_text"] in gold_train_sent_texts)
overlap_dev = sum(1 for r in retained_records if r["cleaned_text"] in gold_dev_sent_texts)

print(f"\n--- Overlap Protection Check ---")
print(f"Overlap with Gold Train: {overlap_train} sentences (tracked)")
print(f"Overlap with Gold Dev:   {overlap_dev} sentences (tracked)")
print("Note: Official Locked Test remains completely sealed and protected.")

# -------------------------------------------------------------
# STEP 4: CORPUS STATISTICS & VOCABULARY GROWTH
# -------------------------------------------------------------
corpus_tokens = [t for r in retained_records for t in r["tokens"]]
corpus_vocab = set(t.lower() for t in corpus_tokens)

new_vocab_tokens = corpus_vocab - gold_train_vocab
vocab_growth_pct = (len(corpus_vocab) - len(gold_train_vocab)) / len(gold_train_vocab) * 100

lengths = [r["token_count"] for r in retained_records]
avg_len = sum(lengths) / len(lengths) if lengths else 0
med_len = float(pd.Series(lengths).median()) if lengths else 0
min_len = min(lengths) if lengths else 0
max_len = max(lengths) if lengths else 0

# Source distribution
source_dist = Counter(r["source"] for r in retained_records)

# Sentences with novel vocabulary
sents_with_new_vocab = sum(1 for r in retained_records if any(t.lower() in new_vocab_tokens for t in r["tokens"]))
pct_sents_new_vocab = sents_with_new_vocab / len(retained_records) * 100 if retained_records else 0

corpus_stats = {
    "retained_sentence_count": len(retained_records),
    "total_token_count": len(corpus_tokens),
    "unique_token_count": len(corpus_vocab),
    "gold_train_vocab_size": len(gold_train_vocab),
    "new_unique_tokens_count": len(new_vocab_tokens),
    "vocab_growth_percentage": vocab_growth_pct,
    "avg_sentence_length": round(avg_len, 2),
    "median_sentence_length": med_len,
    "min_sentence_length": min_len,
    "max_sentence_length": max_len,
    "sentences_with_novel_vocab_count": sents_with_new_vocab,
    "pct_sentences_with_novel_vocab": round(pct_sents_new_vocab, 2),
    "source_distribution": dict(source_dist),
    "quality_exclusion_breakdown": dict(exclusion_counts)
}

print(f"\n--- Corpus Statistics ---")
print(f"Total Tokens:           {len(corpus_tokens):,}")
print(f"Unique Tokens:          {len(corpus_vocab):,} (vs Gold Train: {len(gold_train_vocab):,})")
print(f"New Vocabulary Added:   +{len(new_vocab_tokens):,} words (+{vocab_growth_pct:.1f}% growth)")
print(f"Sentence Length:        Mean={avg_len:.1f}, Median={med_len:.0f}, Min={min_len}, Max={max_len}")
print(f"Sents with Novel Vocab: {sents_with_new_vocab:,} ({pct_sents_new_vocab:.1f}%)")

# -------------------------------------------------------------
# STEP 5: TRANSPARENT ENTITY-CANDIDATE DISCOVERY
# -------------------------------------------------------------
# Heuristic pattern categories
heuristic_triggers = {
    "ORG_INSTITUTION": re.compile(r'\b(?:University|Board|Department|Committee|Corporation|Company|Bank|Hospital|School|College|Institute|Police|Government|Ministry|Channel|News|Bedek|Hukbar|Mission|Union|Federation|Council|Assembly|High Court)\b', re.IGNORECASE),
    "LOC_TOPONYM": re.compile(r'\b(?:Tripura|Agartala|Khumulwng|Udaipur|Dharmanagar|Khowai|Teliamura|Ambassa|Sabroom|Belonia|Kailashahar|Kamalpur|Bishalgarh|Melaghar|Santirbazar|Amarpur|Jampui|Kanchanpur|Twipra|Delhi|Kolkata|Assam|Guwahati|India)\b', re.IGNORECASE),
    "PER_SURNAME_HONORIFIC": re.compile(r'\b(?:Debbarma|Tripura|Jamatia|Reang|Noatia|Chakma|Molsom|Khoroksa|Babu|Prasanta|Maharaja|Bir Bikram|Babulal|Modi|Ambedkar|Gandhi|Nehru)\b', re.IGNORECASE),
    "MISC_EVENT_CULTURE": re.compile(r'\b(?:Christmas|Diwali|Puja|Garia|Ker|Hojagiri|Kharchi|Bwisagu|Mamita|Constitution|Declaration|Pact|Act)\b', re.IGNORECASE),
}

candidate_sentences = []
category_counts = Counter()

for r in retained_records:
    txt = r["cleaned_text"]
    triggers_found = []
    
    for cat_name, reg in heuristic_triggers.items():
        matches = reg.findall(txt)
        if matches:
            triggers_found.append(f"{cat_name}: {', '.join(set(matches[:3]))}")
            category_counts[cat_name] += 1
            
    if triggers_found:
        candidate_sentences.append({
            "sentence_id": r["sentence_id"],
            "source": r["source"],
            "cleaned_text": txt,
            "tokens": r["tokens"],
            "heuristic_triggers": "; ".join(triggers_found)
        })

print(f"\n--- Entity-Candidate Discovery Summary ---")
print(f"Candidate Entity-Bearing Sentences: {len(candidate_sentences):,} ({len(candidate_sentences)/len(retained_records)*100:.1f}% of clean corpus)")
for cat, cnt in category_counts.most_common():
    print(f"  {cat:<25}: {cnt:5,d} occurrences")

# -------------------------------------------------------------
# STEP 6: SAVE ARTIFACTS
# -------------------------------------------------------------
# 1. Cleaned text file
clean_txt_path = PROC_DIR / "kokborok_public_corpus_clean.txt"
with open(clean_txt_path, "w", encoding="utf-8") as f:
    for r in retained_records:
        f.write(r["cleaned_text"] + "\n")
print(f"\nSaved clean corpus text: {clean_txt_path.relative_to(PROJECT_ROOT)}")

# 2. Provenance CSV
prov_csv_path = PROC_DIR / "kokborok_public_corpus_provenance.csv"
with open(prov_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "sentence_id", "source", "source_row_id", "source_file", "license",
        "original_text_hash", "cleaned_text", "exclusion_status", "exclusion_reason"
    ])
    writer.writeheader()
    for r in processed_records:
        writer.writerow({
            "sentence_id": r["sentence_id"],
            "source": r["source"],
            "source_row_id": r["source_row_id"],
            "source_file": r["source_file"],
            "license": r["license"],
            "original_text_hash": r["original_text_hash"],
            "cleaned_text": r["cleaned_text"],
            "exclusion_status": r["exclusion_status"],
            "exclusion_reason": r["exclusion_reason"]
        })
print(f"Saved provenance CSV: {prov_csv_path.relative_to(PROJECT_ROOT)}")

# 3. Candidate CoNLL with explicit UNANNOTATED placeholder format
cand_conll_path = PROC_DIR / "kokborok_ner_candidate_sentences.conll"
with open(cand_conll_path, "w", encoding="utf-8") as f:
    # Header comment documenting that this is raw candidate data
    f.write("# KOKBOROK UNANNOTATED NER CANDIDATE SENTENCES (EXPERIMENT 10 DISCOVERY)\n")
    f.write("# NOTE: All tags are placeholder 'O'. NOT human-annotated gold data.\n\n")
    for item in candidate_sentences:
        f.write(f"# id: {item['sentence_id']} | source: {item['source']} | triggers: {item['heuristic_triggers']}\n")
        for tok in item["tokens"]:
            f.write(f"{tok}\tO\n")
        f.write("\n")
print(f"Saved candidate CoNLL: {cand_conll_path.relative_to(PROJECT_ROOT)}")

# 4. Source inventory CSV
inv_csv_path = REPORTS_DIR / "kokborok_exp10_source_inventory.csv"
with open(inv_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "source_id", "dataset_name", "url", "license", "permitted_use",
        "access_status", "rows_available", "domains", "text_columns", "status"
    ])
    writer.writeheader()
    writer.writerows(source_inventory)
print(f"Saved source inventory CSV: {inv_csv_path.relative_to(PROJECT_ROOT)}")

# 5. Corpus statistics JSON
stat_json_path = REPORTS_DIR / "kokborok_exp10_corpus_statistics.json"
with open(stat_json_path, "w", encoding="utf-8") as f:
    json.dump(corpus_stats, f, indent=2)
print(f"Saved corpus statistics JSON: {stat_json_path.relative_to(PROJECT_ROOT)}")
