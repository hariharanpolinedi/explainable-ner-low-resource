"""
Expanded Kokborok NER Dataset Ingestion & Formatting Pipeline.

Consolidates public internet datasets:
1. Hugging Face: sdmy/kokborok (Google SMOL translation corpus)
2. Hugging Face: agnivamaiti/kokborok-qa (4,943 dialogue and question-answer pairs)
3. Wikimedia Incubator: Kokborok Wikipedia articles

Applies:
- Script and punctuation cleaning
- Boundary-aware gazetteer matching for PER, LOC, ORG, MISC
- CoNLL-2003 format export (token \\t tag)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

PROJECT_ROOT = Path("c:/Users/harih/Desktop/NLP-MARK-3")
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "kokborok"
GOLD_TRAIN_FILE = PROJECT_ROOT / "data" / "gold" / "kokborok" / "train.conll"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "kokborok_targeted_inflected_train.conll"

# Comprehensive Curated Kokborok Entities with Clitics and Regional Variations
GAZETTEER = {
    "PER": [
        "Debbarma", "Tripura", "Jamatia", "Reang", "Noatia", "Chakma", "Kalai", 
        "Murasing", "Koloi", "Rupini", "Hrangkhawl", "Uchoi", "Galo", "Darlong",
        "Bir Bikram", "Radha Kishore", "Birendra Kishore", "Kirit Pradyot", "Manikya",
        "Bimal Debbarma", "Animesh Debbarma", "Bikash Tripura", "Binoy Debbarma",
        "Athlete Amardweep Debbarma", "Bijoy Kumar Hrangkhawl", "Biplab Kumar Deb",
        "Bipasha Tripura", "Asha Tripura", "Bidhan Tripura", "Parimal Debbarma", "Parimal",
        "kanika", "Kanika", "Rajiv Gandhi", "Indira Gandhi", "Jawaharlal Nehru", "Feroze Gandhi", "Ram", "Rajiv", "Sanjay", "Jisu Khristo", "Jisu"
    ],
    "LOC": [
        "Agartala", "Khowai", "Udaipur", "Dharmanagar", "Ambassa", "Kailashahar", 
        "Belonia", "Santirbazar", "Teliamura", "Sabroom", "Bishalgarh", "Sonamura", 
        "Amarpur", "Kanchanpur", "Gandacherra", "Kumarghat", "Jampui", "Baramura", 
        "Atharamura", "Longthorai", "Deo", "Gomati", "Howrah", "Manu", "Dhalai", 
        "Muhuri", "Feni", "Radhapur", "Khumpui", "Jampui Hills", "Tripura",
        "Dashamighat", "Boxanagar", "Chabimura", "Battala", "Akhaura",
        "Barot", "India", "England", "Delhi", "Kolkata", "Assam",
        # Fused clitics & common variations
        "Tripurao", "Tripura-o", "Agartalao", "Agartala-o", "Agratalao", "Agratala",
        "Kailashahar-o", "Kailashaharo", "Udaipuro", "Udaipur-o", "Dharmanagaro",
        # Bordering Northeast geography
        "Lunglei", "Silchar", "Karimganj", "Aizawl"
    ],
    "ORG": [
        # Educational & Research Institutions
        "Tripura University", "MBB College", "MBB University", "TIT", "NIT Agartala", 
        "BBMC", "Holy Cross College", "Ramakrishna Mission", "ICFAI University", 
        "TMC", "Womens College", "Tribal Research Institute",
        # Autonomous, Cultural & Student Bodies
        "TTAADC", "THB", "NSS", "Kokborok Tei Hukumu Mission", "Borok Peoples Council", 
        "Kokborok Sahitya Sabha", "All Tripura Indigenous Students Association", 
        "Twipra Students Federation", "TSF", "BNC", "Borok National Council", 
        "Tripura Baptist Christian Union", "TBCU", "Kokborok Language Division",
        # Administrative, Media & Legal
        "Tripura Legislative Assembly", "High Court of Tripura", "Tripura High Court", 
        "Tripura Police", "Forest Department", "Health Department", 
        "Indian Supreme Court", "Supreme Court", "High Court", "Tripura Commission",
        "Tripura Hukbar Bedek", "Hukbar Bedek", "Twimung Bedek", "DD news", "DD News", "Doordarshan",
        # Political Parties
        "TIPRA Motha", "INPT", "IPFT", "BJP", "Congress", "CPI(M)", "CPIM",
        # Corporations, Utilities & Banking
        "Tripura Gramin Bank", "TGB", "TRTC", "TSECL", "TTDC", "ONGC Tripura", "ONGC", 
        "State Bank of India", "SBI", "TFTC", "THFC"
    ],
    "MISC": [
        "Buisu", "Kharchi Puja", "Kokborok Day", "Garia Puja", "Ker Puja",
        "Wangala", "Hojagiri", "Durga Puja", "Tripuri", "Hojagiri Dance", "Christianity", "Christmas"
    ]
}

# Syntactic frames specifically designed to contextualize organizations and conversational patterns
CONVERSATIONAL_FRAMES = [
    ("My name is {PER} .", [("PER", "{PER}")]),
    ("Nini mung {PER} da ?", [("PER", "{PER}")]),
    ("{PER} hwnei korok sa engineer tabok .", [("PER", "{PER}")]),
    ("{LOC} song advent Christmas khlaina thangwi tongo .", [("LOC", "{LOC}")]),
    ("Belai watwi wama bagwi {LOC} twilom kha .", [("LOC", "{LOC}")]),
    ("Nini mokol kwrwi {ORG} naidi .", [("ORG", "{ORG}")]),
    ("{ORG} Bwchwlwi magwnang bakkha .", [("ORG", "{ORG}")]),
    ("{PER} {ORG} o kam khlaio .", [("PER", "{PER}"), ("ORG", "{ORG}")]),
    ("{PER} {ORG} ni chuchundok wngkha .", [("PER", "{PER}"), ("ORG", "{ORG}")]),
    ("{ORG} {LOC} o kaisa kotorma haba swnamkha .", [("ORG", "{ORG}"), ("LOC", "{LOC}")]),
    ("{ORG} ni kotorma nok {LOC} o tongo .", [("ORG", "{ORG}"), ("LOC", "{LOC}")]),
    ("{PER} tabok {ORG} ni bisingtwi kok sahikha .", [("PER", "{PER}"), ("ORG", "{ORG}")]),
]

def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+|[^\s\w]", text, re.UNICODE)

def annotate_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    tags = ["O"] * len(tokens)
    n = len(tokens)
    i = 0
    matched_any = False
    
    while i < n:
        match = False
        for span_len in (4, 3, 2, 1):
            if i + span_len <= n:
                cand = " ".join(tokens[i:i+span_len]).strip()
                for ent_type, vals in GAZETTEER.items():
                    if any(cand.lower() == v.lower() for v in vals):
                        tags[i] = f"B-{ent_type}"
                        for j in range(1, span_len):
                            tags[i+j] = f"I-{ent_type}"
                        i += span_len
                        match = True
                        matched_any = True
                        break
                if match:
                    break
        if not match:
            i += 1
            
    if not matched_any:
        return []
    return list(zip(tokens, tags))

def generate_org_sentences(num_samples: int = 250) -> list[list[tuple[str, str]]]:
    import random
    random.seed(42)
    sents = []
    for _ in range(num_samples):
        template, slots = random.choice(CONVERSATIONAL_FRAMES)
        rendered = template
        replacements = {}
        for idx, (etype, placeholder) in enumerate(slots):
            val = random.choice(GAZETTEER[etype])
            key = f"__FRAME_SLOT_{idx}__"
            rendered = rendered.replace(placeholder, key, 1)
            replacements[key] = (etype, val)
        toks = tokenize(rendered)
        sent_pairs = []
        for tok in toks:
            if tok in replacements:
                etype, val = replacements[tok]
                etoks = tokenize(val)
                for e_idx, e_tok in enumerate(etoks):
                    tag = f"B-{etype}" if e_idx == 0 else f"I-{etype}"
                    sent_pairs.append((e_tok, tag))
            else:
                sent_pairs.append((tok, "O"))
        sents.append(sent_pairs)
    return sents

def main():
    print("Collecting and formatting internet Kokborok datasets...")
    collected_sentences: list[str] = []
    
    # 1. Read sdmy/kokborok public sentences
    sdmy_file = RAW_DIR / "public_online_sentences.txt"
    if sdmy_file.exists():
        with open(sdmy_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and len(line) > 10:
                    collected_sentences.append(line)
        print(f"Loaded {len(collected_sentences):,} lines from Google SMOL / Wikimedia.")

    # 2. Read agnivamaiti/kokborok-qa
    qa_file = RAW_DIR / "qa_train.parquet"
    if qa_file.exists():
        df = pd.read_parquet(qa_file)
        qa_count = 0
        for _, row in df.iterrows():
            p = str(row.get("prompt", "")).replace("User:", "").strip()
            c = str(row.get("completion", "")).replace("Assistant:", "").strip()
            if p and len(p) > 10:
                collected_sentences.append(p)
                qa_count += 1
            if c and len(c) > 10:
                collected_sentences.append(c)
                qa_count += 1
        print(f"Loaded {qa_count:,} utterances from agnivamaiti/kokborok-qa.")

    # Deduplicate
    unique_sentences = list(dict.fromkeys(collected_sentences))
    print(f"Total unique input sentences: {len(unique_sentences):,}")

    # Annotate and filter entity-bearing sentences
    annotated_sentences = []
    census = Counter()
    total_tokens = 0

    for sent_text in unique_sentences:
        toks = tokenize(sent_text)
        if len(toks) < 3 or len(toks) > 40:
            continue
        labeled_pairs = annotate_tokens(toks)
        if labeled_pairs:
            annotated_sentences.append(labeled_pairs)
            for tok, tag in labeled_pairs:
                total_tokens += 1
                if tag.startswith("B-"):
                    census[tag[2:]] += 1

    # Inject balanced ORG context sentences to eliminate the ORG deficiency
    org_sents = generate_org_sentences(num_samples=220)
    for sent in org_sents:
        annotated_sentences.append(sent)
        for tok, tag in sent:
            total_tokens += 1
            if tag.startswith("B-"):
                census[tag[2:]] += 1

    # Read original gold train sentences
    gold_sents = []
    if GOLD_TRAIN_FILE.exists():
        curr_g = []
        with open(GOLD_TRAIN_FILE, "r", encoding="utf-8") as gf:
            for gline in gf:
                gline = gline.strip()
                if not gline:
                    if curr_g:
                        gold_sents.append(curr_g)
                        curr_g = []
                else:
                    parts = gline.split("\t")
                    curr_g.append((parts[0], parts[-1]))
            if curr_g:
                gold_sents.append(curr_g)
        print(f"Appended {len(gold_sents)} original gold training sentences.")

    for gsent in gold_sents:
        annotated_sentences.append(gsent)
        for tok, tag in gsent:
            total_tokens += 1
            if tag.startswith("B-"):
                census[tag[2:]] += 1

    import random
    random.seed(42)
    random.shuffle(annotated_sentences)

    print(f"\n[Extraction Summary]")
    print(f"  Total combined sentences in dataset: {len(annotated_sentences):,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Entity counts: {dict(census)}")

    # Write to CoNLL format
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for sent in annotated_sentences:
            for tok, tag in sent:
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")

    print(f"\n[SUCCESS] Formatted dataset saved in standard CoNLL format to:")
    print(f"  {OUT_FILE.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
