"""
Experiment 2: Balanced & Clitic-Aware Kokborok Dataset Generator.

Interventions for Recall Improvement:
1. Coverage Guarantee: Every entity in GAZETTEER is represented at least 12-15 times across diverse syntactic frames.
2. Clitic & Inflectional Realism:
   - Standard Kokborok locative clitic '-o' (e.g. 'Tripurao', 'Tripura-o', 'Kailashahar-o', 'Agartalao', 'Udaipuro').
   - Kokborok genitive clitic 'ni' / '-ni'.
3. Casing Robustness:
   - Controlled lowercase variants for informal conversational text ('kanika', 'dd news', 'parimal debbarma').
4. Multi-Entity Sentences:
   - Sentences containing combinations of PER, ORG, and LOC to improve multi-entity context modeling.
5. Strict Independence & Zero-Leakage:
   - Independent public text from sdmy/kokborok and agnivamaiti/kokborok-qa.
   - Never copies sentences from gold dev.conll or locked test.conll.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("c:/Users/harih/Desktop/NLP-MARK-3")
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "kokborok"
GOLD_TRAIN_FILE = PROJECT_ROOT / "data" / "gold" / "kokborok" / "train.conll"
GOLD_DEV_FILE = PROJECT_ROOT / "data" / "gold" / "kokborok" / "dev.conll"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "kokborok_exp2_balanced_train.conll"

# Comprehensive Curated Kokborok Entities
GAZETTEER = {
    "PER": [
        "Debbarma", "Tripura", "Jamatia", "Reang", "Noatia", "Chakma", "Kalai", 
        "Murasing", "Koloi", "Rupini", "Hrangkhawl", "Uchoi", "Galo", "Darlong",
        "Bir Bikram", "Radha Kishore", "Birendra Kishore", "Kirit Pradyot", "Manikya",
        "Bimal Debbarma", "Animesh Debbarma", "Bikash Tripura", "Binoy Debbarma",
        "Athlete Amardweep Debbarma", "Bijoy Kumar Hrangkhawl", "Biplab Kumar Deb",
        "Bipasha Tripura", "Asha Tripura", "Bidhan Tripura", "Parimal Debbarma", "Parimal",
        "Kanika", "kanika", "Rajiv Gandhi", "Indira Gandhi", "Jawaharlal Nehru", "Feroze Gandhi",
        "Ram", "Rajiv", "Sanjay", "Jisu Khristo", "Jisu"
    ],
    "LOC": [
        "Agartala", "Khowai", "Udaipur", "Dharmanagar", "Ambassa", "Kailashahar", 
        "Belonia", "Santirbazar", "Teliamura", "Sabroom", "Bishalgarh", "Sonamura", 
        "Amarpur", "Kanchanpur", "Gandacherra", "Kumarghat", "Jampui", "Baramura", 
        "Atharamura", "Longthorai", "Deo", "Gomati", "Howrah", "Manu", "Dhalai", 
        "Muhuri", "Feni", "Radhapur", "Khumpui", "Jampui Hills", "Tripura",
        "Dashamighat", "Boxanagar", "Chabimura", "Battala", "Akhaura",
        "Barot", "India", "England", "Delhi", "Kolkata", "Assam",
        "Tripurao", "Tripura-o", "Agartalao", "Agartala-o", "Agratalao", "Agratala",
        "Kailashahar-o", "Kailashaharo", "Udaipuro", "Udaipur-o", "Dharmanagaro",
        "Lunglei", "Silchar", "Karimganj", "Aizawl"
    ],
    "ORG": [
        "Tripura University", "MBB College", "MBB University", "TIT", "NIT Agartala", 
        "BBMC", "Holy Cross College", "Ramakrishna Mission", "ICFAI University", 
        "TMC", "Womens College", "Tribal Research Institute",
        "TTAADC", "THB", "NSS", "Kokborok Tei Hukumu Mission", "Borok Peoples Council", 
        "Kokborok Sahitya Sabha", "All Tripura Indigenous Students Association", 
        "Twipra Students Federation", "TSF", "BNC", "Borok National Council", 
        "Tripura Baptist Christian Union", "TBCU", "Kokborok Language Division",
        "Tripura Legislative Assembly", "High Court of Tripura", "Tripura High Court", 
        "Tripura Police", "Forest Department", "Health Department", 
        "Indian Supreme Court", "Supreme Court", "High Court", "Tripura Commission",
        "Tripura Hukbar Bedek", "Hukbar Bedek", "Twimung Bedek",
        "DD news", "DD News", "Doordarshan",
        "TIPRA Motha", "INPT", "IPFT", "BJP", "Congress", "CPI(M)", "CPIM",
        "Tripura Gramin Bank", "TGB", "TRTC", "TSECL", "TTDC", "ONGC Tripura", "ONGC", 
        "State Bank of India", "SBI", "TFTC", "THFC"
    ],
    "MISC": [
        "Buisu", "Kharchi Puja", "Kokborok Day", "Garia Puja", "Ker Puja",
        "Wangala", "Hojagiri", "Durga Puja", "Tripuri", "Hojagiri Dance", "Christianity", "Christmas"
    ]
}

SYNTACTIC_FRAMES = [
    # PER frames
    ("My name is {PER} .", [("PER", "{PER}")]),
    ("Nini mung {PER} da ?", [("PER", "{PER}")]),
    ("{PER} hwnei korok sa engineer tabok .", [("PER", "{PER}")]),
    ("{PER} kok sahikha tei phaikha .", [("PER", "{PER}")]),
    ("Ang {PER} no nuk-kha tabok .", [("PER", "{PER}")]),
    # LOC frames (nominative and clitic-bearing)
    ("{LOC} song advent Christmas khlaina thangwi tongo .", [("LOC", "{LOC}")]),
    ("Belai watwi wama bagwi {LOC} twilom kha .", [("LOC", "{LOC}")]),
    ("Bo song {LOC} o tongo .", [("LOC", "{LOC}")]),
    ("Abo {LOC} ni kotorma nok .", [("LOC", "{LOC}")]),
    ("Ang tabok {LOC} thangna mwno .", [("LOC", "{LOC}")]),
    ("{LOC} ni luku rok kaham tongo .", [("LOC", "{LOC}")]),
    # ORG frames
    ("Nini mokol kwrwi {ORG} naidi .", [("ORG", "{ORG}")]),
    ("{ORG} Bwchwlwi magwnang bakkha .", [("ORG", "{ORG}")]),
    ("{ORG} kaisa kotorma haba swnamkha .", [("ORG", "{ORG}")]),
    ("Tabok {ORG} ni kokno sahikha .", [("ORG", "{ORG}")]),
    # Multi-entity frames (combining PER, LOC, ORG, MISC)
    ("{PER} {ORG} o kam khlaio .", [("PER", "{PER}"), ("ORG", "{ORG}")]),
    ("{PER} {ORG} ni chuchundok wngkha .", [("PER", "{PER}"), ("ORG", "{ORG}")]),
    ("{ORG} {LOC} o kaisa kotorma haba swnamkha .", [("ORG", "{ORG}"), ("LOC", "{LOC}")]),
    ("{ORG} ni kotorma nok {LOC} o tongo .", [("ORG", "{ORG}"), ("LOC", "{LOC}")]),
    ("{PER} tabok {ORG} ni bisingtwi {LOC} thangna mwno .", [("PER", "{PER}"), ("ORG", "{ORG}"), ("LOC", "{LOC}")]),
    ("{LOC} o borok rok {MISC} khlaio .", [("LOC", "{LOC}"), ("MISC", "{MISC}")]),
    ("{PER} {LOC} o thangwi {MISC} sal khlaikha .", [("PER", "{PER}"), ("LOC", "{LOC}"), ("MISC", "{MISC}")]),
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

def generate_balanced_entity_sentences(target_per_entity: int = 14) -> list[list[tuple[str, str]]]:
    """
    Generates balanced syntactic frame examples so that EVERY entity in the GAZETTEER
    receives at least target_per_entity occurrences.
    """
    random.seed(42)
    sents = []
    
    for etype, ent_list in GAZETTEER.items():
        # Match frames that accept this entity type
        valid_frames = [f for f in SYNTACTIC_FRAMES if any(slot[0] == etype for slot in f[1])]
        if not valid_frames:
            continue
        
        for ent_val in ent_list:
            for rep in range(target_per_entity):
                template, slots = random.choice(valid_frames)
                rendered = template
                replacements = {}
                for idx, (slot_etype, placeholder) in enumerate(slots):
                    if slot_etype == etype:
                        val = ent_val
                    else:
                        val = random.choice(GAZETTEER[slot_etype])
                    key = f"__SLOT_{idx}__"
                    rendered = rendered.replace(placeholder, key, 1)
                    replacements[key] = (slot_etype, val)
                
                toks = tokenize(rendered)
                sent_pairs = []
                for tok in toks:
                    if tok in replacements:
                        sel_etype, sel_val = replacements[tok]
                        etoks = tokenize(sel_val)
                        for e_idx, e_tok in enumerate(etoks):
                            tag = f"B-{sel_etype}" if e_idx == 0 else f"I-{sel_etype}"
                            sent_pairs.append((e_tok, tag))
                    else:
                        sent_pairs.append((tok, "O"))
                sents.append(sent_pairs)
    return sents

def main():
    print("Building Experiment 2 Balanced Kokborok Dataset...")
    random.seed(42)
    
    # 1. Read Raw Public Sentences (Google SMOL / Wikimedia)
    collected_sentences: list[str] = []
    sdmy_file = RAW_DIR / "public_online_sentences.txt"
    if sdmy_file.exists():
        with open(sdmy_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and len(line) > 10:
                    collected_sentences.append(line)
        print(f"Loaded {len(collected_sentences):,} lines from public online corpus.")

    # 2. Read QA pairs (agnivamaiti/kokborok-qa)
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
        print(f"Loaded {qa_count:,} utterances from Kokborok QA.")

    # 3. Deduplicate
    unique_sentences = list(dict.fromkeys(collected_sentences))
    print(f"Total unique input sentences: {len(unique_sentences):,}")

    # 4. Annotate natural corpus
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

    print(f"Natural corpus entity sentences: {len(annotated_sentences):,} | Entities: {dict(census)}")

    # 5. Inject Balanced Syntactic Frame Sentences
    balanced_frames = generate_balanced_entity_sentences(target_per_entity=14)
    print(f"Generated {len(balanced_frames):,} balanced syntactic frame sentences.")
    for sent in balanced_frames:
        annotated_sentences.append(sent)
        for tok, tag in sent:
            total_tokens += 1
            if tag.startswith("B-"):
                census[tag[2:]] += 1

    # 6. Include Gold Train sentences
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
        print(f"Appended {len(gold_sents):,} original gold training sentences.")

    for gsent in gold_sents:
        annotated_sentences.append(gsent)
        for tok, tag in gsent:
            total_tokens += 1
            if tag.startswith("B-"):
                census[tag[2:]] += 1

    # Check that NO dev sentence is included
    dev_sents_str = set()
    if GOLD_DEV_FILE.exists():
        curr_d = []
        with open(GOLD_DEV_FILE, "r", encoding="utf-8") as df:
            for dline in df:
                dline = dline.strip()
                if not dline:
                    if curr_d:
                        dev_sents_str.add(" ".join(w for w, _ in curr_d))
                        curr_d = []
                else:
                    parts = dline.split()
                    curr_d.append((parts[0], parts[-1]))
            if curr_d:
                dev_sents_str.add(" ".join(w for w, _ in curr_d))

    # Filter out any accidental dev matches
    final_sentences = []
    filtered_dev_matches = 0
    for s in annotated_sentences:
        s_str = " ".join(w for w, _ in s)
        if s_str in dev_sents_str:
            filtered_dev_matches += 1
            continue
        final_sentences.append(s)

    print(f"Filtered dev matches (strictly protected): {filtered_dev_matches}")
    random.shuffle(final_sentences)

    print(f"\n[Final Dataset Summary]")
    print(f"  Total sentences: {len(final_sentences):,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Entity counts: {dict(census)}")

    # Write to CoNLL
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for sent in final_sentences:
            for tok, tag in sent:
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")

    print(f"[SUCCESS] Saved to {OUT_FILE.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
