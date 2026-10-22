"""
Comprehensive Training Dataset Generator for Kokborok NER.
Combines:
1. Gold Kokborok annotations (data/gold/kokborok/train.conll)
2. Public Silver dataset from Hugging Face & Wikimedia (data/silver/kokborok/train.conll)
3. Authentic Kokborok syntactic frames with authentic Tripura entities
   (Kings, Leaders, TTAADC, Tipra Motha, Khumulwng, Buisu, Kokborok Day, etc.)

Strictly zero pretrained models or weights.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_TRAIN = PROJECT_ROOT / "data" / "gold" / "kokborok" / "train.conll"
SILVER_TRAIN = PROJECT_ROOT / "data" / "silver" / "kokborok" / "train.conll"
OUT_TRAIN = PROJECT_ROOT / "data" / "processed" / "kokborok_neural_learning_train.conll"
OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Authentic Kokborok Named Entities
# ---------------------------------------------------------------------------
KOKBOROK_PERSONS = [
    "Maharaja Bir Bikram Kishore Manikya",
    "Maharaja Bir Bikram",
    "Bir Bikram Kishore Manikya",
    "Maharaja Radha Kishore Manikya",
    "Radha Kishore Manikya",
    "Rabindranath Tagore",
    "Droupadi Murmu",
    "Pradyot Bikram Manikya Debbarma",
    "Pradyot Kishore Debbarma",
    "Kirit Bikram Kishore Manikya",
    "Dasarath Deb",
    "Birendra Kishore Manikya",
    "Alindra Debbarma",
    "Naphurai Jamatia",
    "Mevar Kumar Jamatia",
    "Manik Saha",
    "Biplab Kumar Deb",
    "Jitendra Chaudhury",
    "Rabindra Debbarma",
    "Animesh Debbarma",
    "Parimal Debbarma",
    "Sukanta Debbarma",
    "Ratan Debbarma",
    "Narendra Modi",
    "Pramod Reang",
    "Chandra Jamatia",
]

KOKBOROK_PLACES = [
    "Tripura", "Agartala", "Khumulwng", "Udaipur", "Dharmanagar",
    "Khowai", "Ambassa", "Belonia", "Sabroom", "Suryamaninagar",
    "Bishalgarh", "Santirbazar", "Jampui Hills", "Howrah", "Gomati",
    "New Delhi", "India", "Bangladesh", "Guwahati", "Kolkata",
    "Lunglei", "Kalyanpur", "Jirania", "Bodhjungnagar", "Melaghar",
    "Gorkhabasti", "Old Agartala", "Rudrasagar Lake", "Gomati River"
]

KOKBOROK_ORGS = [
    "Tripura Tribal Areas Autonomous District Council",
    "TTAADC",
    "Tipra Motha",
    "Tripura Hukbar Bedek",
    "Hukbar Bedek",
    "DD news",
    "Doordarshan",
    "Tripura University",
    "National Institute of Technology Agartala",
    "National Institute of Technology",
    "Indian Institute of Information Technology",
    "Maharaja Bir Bikram University",
    "MBB University",
    "Maharaja Bir Bikram College",
    "Tripura Medical College",
    "Twipra Students Federation",
    "TSF",
    "Kokborok Sahitya Sabha",
    "Kokborok Tei Hukumu Mission",
    "Tripura Legislative Assembly",
    "Government of Tripura",
    "Ministry of Education",
    "Press Trust of India",
    "Rubber Board of India",
    "Tripura State Museum",
    "Tata Technologies",
    "State Bank of India",
    "ONGC Tripura"
]

KOKBOROK_EVENTS = [
    "Buisu",
    "Kharchi Puja",
    "Kokborok Day",
    "Garia Puja",
    "Ker Puja",
    "Neermahal Water Festival",
    "Durga Puja",
    "Diwali",
    "Christmas"
]

KOKBOROK_FACILITIES = [
    "Ujjayanta Palace",
    "Neermahal",
    "Maharaja Bir Bikram Airport",
    "MBB Airport",
    "Maharaja Bir Bikram Stadium",
    "Agartala Railway Station",
    "Tripura Sundari Temple",
    "Chaturdash Devata Bari"
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


def tag_span(text: str, label: str) -> list[tuple[str, str]]:
    tokens = text.split()
    res = []
    for i, tok in enumerate(tokens):
        prefix = "B-" if i == 0 else "I-"
        res.append((tok, f"{prefix}{label}"))
    return res


def tag_span_with_suffix(text: str, label: str, suffix: str) -> list[tuple[str, str]]:
    tokens = text.split()
    res = []
    for i, tok in enumerate(tokens):
        prefix = "B-" if i == 0 else "I-"
        if i == len(tokens) - 1:
            res.append((f"{tok}{suffix}", f"{prefix}{label}"))
        else:
            res.append((tok, f"{prefix}{label}"))
    return res


def generate_kokborok_template(idx: int) -> list[tuple[str, str]]:
    t_type = idx % 8
    tokens = []

    per = random.choice(KOKBOROK_PERSONS)
    loc = random.choice(KOKBOROK_PLACES)
    loc2 = random.choice([l for l in KOKBOROK_PLACES if l != loc])
    org = random.choice(KOKBOROK_ORGS)
    event = random.choice(KOKBOROK_EVENTS)
    fac = random.choice(KOKBOROK_FACILITIES)
    month = random.choice(MONTHS)
    year = random.choice(["1979", "1949", "2021", "2023", "2024", "2025"])
    day = str(random.randint(1, 28))

    if t_type == 0:
        # Template 1: [PER] [LOC]-ni [LOC2]-o [FAC] snammani .
        tokens.extend(tag_span(per, "PER"))
        tokens.extend(tag_span_with_suffix(loc, "LOC", "-ni"))
        tokens.extend(tag_span_with_suffix(loc2, "LOC", "-o"))
        tokens.extend(tag_span(fac, "LOC"))
        tokens.extend([("snammani", "O"), (".", "O")])

    elif t_type == 1:
        # Template 2: Tabuk , [ORG]-ni pande wngkha [LOC] .
        tokens.extend([("Tabuk", "O"), (",", "O")])
        tokens.extend(tag_span_with_suffix(org, "ORG", "-ni"))
        tokens.extend([("pande", "O"), ("wngkha", "O")])
        tokens.extend(tag_span(loc, "LOC"))
        tokens.extend([(".", "O")])

    elif t_type == 2:
        # Template 3: [EVENT] romwi [MONTH] [DAY] , [YEAR] bisi-o chengmani .
        tokens.extend(tag_span(event, "MISC"))
        tokens.extend([("romwi", "O")])
        tokens.extend(tag_span(month, "MISC"))
        tokens.append((day, "I-MISC"))
        tokens.append((",", "O"))
        tokens.append((year, "B-MISC"))
        tokens.extend([("bisi-o", "O"), ("chengmani", "O"), (".", "O")])

    elif t_type == 3:
        # Template 4: [PER] , [ORG]-ni padhai , [LOC]-o kothoma sanlaikha .
        tokens.extend(tag_span(per, "PER"))
        tokens.append((",", "O"))
        tokens.extend(tag_span_with_suffix(org, "ORG", "-ni"))
        tokens.extend([("padhai", "O"), (",", "O")])
        tokens.extend(tag_span_with_suffix(loc, "LOC", "-o"))
        tokens.extend([("kothoma", "O"), ("sanlaikha", "O"), (".", "O")])

    elif t_type == 4:
        # Template 5: [LOC]-ni borokrog [MONTH] tal-o [EVENT] twi rwng kthar khwlai-o .
        tokens.extend(tag_span_with_suffix(loc, "LOC", "-ni"))
        tokens.extend([("borokrog", "O")])
        tokens.extend(tag_span(month, "MISC"))
        tokens.extend([("tal-o", "O")])
        tokens.extend(tag_span(event, "MISC"))
        tokens.extend([("twi", "O"), ("rwng", "O"), ("kthar", "O"), ("khwlai-o", "O"), (".", "O")])

    elif t_type == 5:
        # Template 6: [ORG] [LOC]-o tongo .
        tokens.extend(tag_span(org, "ORG"))
        tokens.extend(tag_span_with_suffix(loc, "LOC", "-o"))
        tokens.extend([("tongo", "O"), (".", "O")])

    elif t_type == 6:
        # Template 7: [PER] wngkha [LOC]-ni mantri .
        tokens.extend(tag_span(per, "PER"))
        tokens.extend([("wngkha", "O")])
        tokens.extend(tag_span_with_suffix(loc, "LOC", "-ni"))
        tokens.extend([("mantri", "O"), (".", "O")])

    elif t_type == 7:
        # Template 8: [LOC] wngkha [LOC2] ni pande .
        tokens.extend(tag_span(loc, "LOC"))
        tokens.extend([("wngkha", "O")])
        tokens.extend(tag_span_with_suffix(loc2, "LOC", "-ni"))
        tokens.extend([("pande", "O"), (".", "O")])

    elif t_type == 8:
        # Template 9: [ORG] tei [ORG2] [LOC]-o office tongo .
        org2 = random.choice([o for o in KOKBOROK_ORGS if o != org])
        tokens.extend(tag_span(org, "ORG"))
        tokens.extend([("tei", "O")])
        tokens.extend(tag_span(org2, "ORG"))
        tokens.extend(tag_span_with_suffix(loc, "LOC", "-o"))
        tokens.extend([("office", "O"), ("tongo", "O"), (".", "O")])

    elif t_type == 9:
        # Template 10: [PER] [YEAR] bisi-o [FAC] snammani .
        tokens.extend(tag_span(per, "PER"))
        tokens.append((year, "B-MISC"))
        tokens.extend([("bisi-o", "O")])
        tokens.extend(tag_span(fac, "LOC"))
        tokens.extend([("snammani", "O"), (".", "O")])

    elif t_type == 10:
        # Template 11: [FAC] [LOC]-ni bising-o tongo .
        tokens.extend(tag_span(fac, "LOC"))
        tokens.extend(tag_span_with_suffix(loc, "LOC", "-ni"))
        tokens.extend([("bising-o", "O"), ("tongo", "O"), (".", "O")])

    else:
        # Template 12: [PER] [MONTH] tal-o [LOC]-ni [LOC2]-o phaikha .
        tokens.extend(tag_span(per, "PER"))
        tokens.extend(tag_span(month, "MISC"))
        tokens.extend([("tal-o", "O")])
        tokens.extend(tag_span_with_suffix(loc, "LOC", "-ni"))
        tokens.extend(tag_span_with_suffix(loc2, "LOC", "-o"))
        tokens.extend([("phaikha", "O"), (".", "O")])

    return tokens


def main() -> None:
    random.seed(42)
    sentences: list[list[tuple[str, str]]] = []

    # 1. Read Gold Train
    if GOLD_TRAIN.is_file():
        with open(GOLD_TRAIN, "r", encoding="utf-8") as f:
            curr = []
            for line in f:
                line = line.strip()
                if not line:
                    if curr:
                        sentences.append(curr)
                        curr = []
                else:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        curr.append((parts[0], parts[-1]))
            if curr:
                sentences.append(curr)
        print(f"Loaded {len(sentences)} Gold training sentences.")

    # 2. Read Public Silver Train
    if SILVER_TRAIN.is_file():
        silv_count = 0
        with open(SILVER_TRAIN, "r", encoding="utf-8") as f:
            curr = []
            for line in f:
                line = line.strip()
                if not line:
                    if curr:
                        sentences.append(curr)
                        curr = []
                        silv_count += 1
                else:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        curr.append((parts[0], parts[-1]))
            if curr:
                sentences.append(curr)
                silv_count += 1
        print(f"Loaded {silv_count} Public Silver training sentences.")

    # 3. Generate 3,000 authentic Kokborok syntactic sentences
    n_synth = 3000
    for i in range(n_synth):
        sentences.append(generate_kokborok_template(i))

    print(f"Total training dataset with Silver + Authentic Generalization: {len(sentences):,} sentences.")

    # 4. Save combined dataset
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for sent in sentences:
            for w, tag in sent:
                f.write(f"{w}\t{tag}\n")
            f.write("\n")

    print(f"[SUCCESS] Saved combined training data to: {OUT_TRAIN.relative_to(PROJECT_ROOT)} ({OUT_TRAIN.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
