"""
Kokborok Entity-Dense Silver Dataset Generator.

Solves the 'Outside-Drowning' problem:
1. Curates balanced entity mentions for PER, LOC, ORG, and MISC.
2. Extracts entity-bearing sentences from data/raw/kokborok/public_online_sentences.txt.
3. Injects high-quality Kokborok syntactic frames to balance underrepresented classes (PER, ORG).
4. Combines with data/gold/kokborok/train.conll.
"""

from __future__ import annotations

import random
from pathlib import Path
import re

PROJECT_ROOT = Path("c:/Users/harih/Desktop/NLP-MARK-3")
RAW_SENTENCES = PROJECT_ROOT / "data" / "raw" / "kokborok" / "public_online_sentences.txt"
GOLD_TRAIN = PROJECT_ROOT / "data" / "gold" / "kokborok" / "train.conll"
OUTPUT_CONLL = PROJECT_ROOT / "data" / "processed" / "kokborok_balanced_dense_train.conll"

# Comprehensive Curated Kokborok Entities
GAZETTEER = {
    "PER": [
        "Debbarma", "Tripura", "Jamatia", "Reang", "Noatia", "Chakma", "Kalai", 
        "Murasing", "Koloi", "Rupini", "Hrangkhawl", "Uchoi", "Galo", "Darlong",
        "Bir Bikram", "Radha Kishore", "Birendra Kishore", "Kirit Pradyot", "Manikya",
        "Bimal Debbarma", "Animesh Debbarma", "Bikash Tripura", "Binoy Debbarma",
        "Athlete Amardweep Debbarma", "Bijoy Kumar Hrangkhawl", "Biplab Kumar Deb",
        "Bipasha Tripura", "Asha Tripura", "Bidhan Tripura",
        "Rajiv Gandhi", "Indira Gandhi", "Jawaharlal Nehru", "Feroze Gandhi", "Ram", "Rajiv", "Sanjay"
    ],
    "LOC": [
        "Agartala", "Khowai", "Udaipur", "Dharmanagar", "Ambassa", "Kailashahar", 
        "Belonia", "Santirbazar", "Teliamura", "Sabroom", "Bishalgarh", "Sonamura", 
        "Amarpur", "Kanchanpur", "Gandacherra", "Kumarghat", "Jampui", "Baramura", 
        "Atharamura", "Longthorai", "Deo", "Gomati", "Howrah", "Manu", "Dhalai", 
        "Muhuri", "Feni", "Radhapur", "Khumpui", "Jampui Hills", "Tripura",
        "Dashamighat", "Boxanagar", "Chabimura", "Battala", "Akhaura",
        "Barot", "India", "England", "Delhi", "Kolkata", "Assam"
    ],
    "ORG": [
        "TTAADC", "THB", "NSS", "Tripura University", "MBB College", "TIT", 
        "Kokborok Tei Hukumu Mission", "Borok Peoples Council", "Kokborok Sahitya Sabha",
        "Tripura Legislative Assembly", "High Court of Tripura", "THFC", "TFTC",
        "All Tripura Indigenous Students Association", "Kokborok Language Division",
        "Indian Supreme Court", "Supreme Court", "High Court", "Congress"
    ],
    "MISC": [
        "Buisu", "Kharchi Puja", "Kokborok Day", "Garia Puja", "Ker Puja",
        "Wangala", "Hojagiri", "Durga Puja", "Tripuri", "Hojagiri Dance"
    ]
}

# Authentic Kokborok syntactic frames for entity contextualization
SYNTACTIC_FRAMES = [
    # PER + LOC
    ("{PER} {LOC} o thangkha .", [("PER", "{PER}"), ("LOC", "{LOC}")]),
    ("{PER} tabok {LOC} o tongo .", [("PER", "{PER}"), ("LOC", "{LOC}")]),
    ("{PER} ni nok {LOC} o tongo .", [("PER", "{PER}"), ("LOC", "{LOC}")]),
    ("{PER} tei {PER} {LOC} o phaikha .", [("PER", "{PER}"), ("PER", "{PER}"), ("LOC", "{LOC}")]),
    
    # PER + ORG
    ("{PER} {ORG} ni kaisa dophani borok .", [("PER", "{PER}"), ("ORG", "{ORG}")]),
    ("{PER} {ORG} o kam khlaio .", [("PER", "{PER}"), ("ORG", "{ORG}")]),
    ("{ORG} ni bisingtwi {PER} kok sahikha .", [("ORG", "{ORG}"), ("PER", "{PER}")]),
    ("{PER} {ORG} ni chuchundok wngkha .", [("PER", "{PER}"), ("ORG", "{ORG}")]),
    
    # ORG + LOC
    ("{ORG} ni kotorma nok {LOC} o tongo .", [("ORG", "{ORG}"), ("LOC", "{LOC}")]),
    ("{ORG} {LOC} o kaisa kotorma mwsamung khlaikha .", [("ORG", "{ORG}"), ("LOC", "{LOC}")]),
    
    # MISC + LOC
    ("{MISC} salo {LOC} o belai haba kotorma wngkha .", [("MISC", "{MISC}"), ("LOC", "{LOC}")]),
    ("{LOC} ni borokrog {MISC} maniyo .", [("LOC", "{LOC}"), ("MISC", "{MISC}")]),
    
    # Single Entities in context
    ("{PER} belai kotor borok .", [("PER", "{PER}")]),
    ("{ORG} kotor kaisa tongkhorok .", [("ORG", "{ORG}")]),
    ("{LOC} belai naithok thaio .", [("LOC", "{LOC}")]),
    ("{MISC} chini kotor sal .", [("MISC", "{MISC}")]),
]

def tokenize_simple(text: str) -> list[str]:
    # Match words or isolated punctuation
    return re.findall(r"[\w]+|[^\s\w]", text, re.UNICODE)

def generate_frame_sentences(num_samples: int = 400) -> list[list[tuple[str, str]]]:
    random.seed(42)
    sentences = []
    
    for _ in range(num_samples):
        template, entity_slots = random.choice(SYNTACTIC_FRAMES)
        rendered_text = template
        replacements = {}
        
        # Pick random entities for the slots
        for idx, (ent_type, placeholder) in enumerate(entity_slots):
            ent_val = random.choice(GAZETTEER[ent_type])
            key = f"__ENT_{idx}__"
            # Replace first occurrence of placeholder
            rendered_text = rendered_text.replace(placeholder, key, 1)
            replacements[key] = (ent_type, ent_val)
            
        # Now reconstruct tokens with tags
        tokens = tokenize_simple(rendered_text)
        sent_tags = []
        for tok in tokens:
            if tok in replacements:
                ent_type, ent_val = replacements[tok]
                ent_tokens = tokenize_simple(ent_val)
                for e_idx, e_tok in enumerate(ent_tokens):
                    tag = f"B-{ent_type}" if e_idx == 0 else f"I-{ent_type}"
                    sent_tags.append((e_tok, tag))
            else:
                sent_tags.append((tok, "O"))
        sentences.append(sent_tags)
        
    return sentences

def extract_from_raw_corpus(max_sentences: int = 300) -> list[list[tuple[str, str]]]:
    if not RAW_SENTENCES.exists():
        return []
        
    extracted = []
    with open(RAW_SENTENCES, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    for line in lines:
        tokens = tokenize_simple(line)
        if len(tokens) < 4 or len(tokens) > 35:
            continue
            
        # Match gazetteer
        tags = ["O"] * len(tokens)
        matched_any = False
        
        n = len(tokens)
        i = 0
        while i < n:
            match = False
            for span_len in (3, 2, 1):
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
                
        if matched_any:
            extracted.append(list(zip(tokens, tags)))
            if len(extracted) >= max_sentences:
                break
                
    return extracted

def main():
    print("Building balanced Kokborok training dataset...")
    
    # 1. Generate frame-based sentences for class balance
    frame_sents = generate_frame_sentences(num_samples=450)
    print(f"Generated {len(frame_sents)} entity-balanced synthetic frame sentences.")
    
    # 2. Extract entity-bearing sentences from raw corpus
    corpus_sents = extract_from_raw_corpus(max_sentences=250)
    print(f"Extracted {len(corpus_sents)} entity-bearing natural sentences from raw corpus.")
    
    # 3. Read original gold train
    gold_sents = []
    with open(GOLD_TRAIN, "r", encoding="utf-8") as f:
        curr = []
        for line in f:
            line = line.strip()
            if not line:
                if curr:
                    gold_sents.append(curr)
                    curr = []
            else:
                parts = line.split("\t")
                tok = parts[0]
                tag = parts[1] if len(parts) > 1 else parts[-1]
                curr.append((tok, tag))
        if curr:
            gold_sents.append(curr)
    print(f"Loaded {len(gold_sents)} sentences from original gold train.")
    
    # 4. Merge all and write
    all_sents = gold_sents + frame_sents + corpus_sents
    random.seed(42)
    random.shuffle(all_sents)
    
    OUTPUT_CONLL.parent.mkdir(parents=True, exist_ok=True)
    entity_counts = {}
    with open(OUTPUT_CONLL, "w", encoding="utf-8") as out:
        for sent in all_sents:
            for tok, tag in sent:
                out.write(f"{tok}\t{tag}\n")
                if tag.startswith("B-"):
                    etype = tag[2:]
                    entity_counts[etype] = entity_counts.get(etype, 0) + 1
            out.write("\n")
            
    print(f"Successfully saved {len(all_sents)} balanced sentences to {OUTPUT_CONLL}")
    print(f"Entity census in new training pool: {entity_counts}")

if __name__ == "__main__":
    main()
