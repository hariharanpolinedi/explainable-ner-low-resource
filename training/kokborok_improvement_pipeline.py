"""
Kokborok NER Improvement Pipeline — Complete End-to-End.

Phases:
  1. Expand gazetteers with real Tripura/Kokborok entities from web research
  2. Extract sentences from all 6 Excel datasets + auto-tag via gazetteer
  3. Generate 5,000+ diverse synthetic sentences from 40 templates
  4. Apply entity-substitution + suffix augmentation (×8 per entity sentence)
  5. Retrain with Option-C protected backbone (20 epochs)
  6. Evaluate on locked gold-test with per-entity breakdown

Strictly zero pretrained models — 100% from-scratch pipeline.
"""

from __future__ import annotations

import copy
import json
import random
import sys

# Fix Windows encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# PHASE 1: MASSIVELY EXPANDED GAZETTEERS (from web research + existing data)
# ============================================================================

PERSONS_EXPANDED = [
    # Existing gazetteer persons
    "Maharaja Bir Bikram Kishore Manikya", "Maharaja Bir Bikram", "Bir Bikram Kishore Manikya",
    "Maharaja Radha Kishore Manikya", "Radha Kishore Manikya",
    "Pradyot Bikram Manikya Debbarma", "Pradyot Kishore Debbarma",
    "Kirit Bikram Kishore Manikya", "Dasarath Deb", "Birendra Kishore Manikya",
    "Alindra Debbarma", "Naphurai Jamatia", "Mevar Kumar Jamatia",
    "Manik Saha", "Biplab Kumar Deb", "Jitendra Chaudhury",
    "Rabindra Debbarma", "Animesh Debbarma", "Parimal Debbarma",
    "Sukanta Debbarma", "Ratan Debbarma", "Narendra Modi",
    "Pramod Reang", "Chandra Jamatia",
    # NEW: From web research — famous Tripuri people
    "Dipa Karmakar", "Somdev Devvarman", "Sachin Dev Burman", "Rahul Dev Burman",
    "Sourabhee Debbarma", "Droupadi Murmu", "Rabindranath Tagore",
    "Thakur Radhamohan Debbarma", "Chandra Kanta Murasingh", "Bikashrai Debbarma",
    "Sukhendu Debbarma", "Bijoy Debbarma", "Nanda Kumar Debbarma",
    "Nagendra Jamatia", "Narendra Deb Barma", "Sefali Debbarma",
    "Manik Sarkar", "Debashis Jamatia", "Rina Debbarma",
    "Anupam Chakma", "Suman Reang", "Michael Das",
    # NEW: Common Tripuri / Bengali names for diversity
    "Rajesh Debbarma", "Sunita Jamatia", "Bikram Reang", "Mohan Tripura",
    "Rahul Debbarma", "Priya Chakma", "Kamal Koloi", "Ranjit Murasing",
    "Suresh Noatia", "Lakshmi Debbarma", "Anil Jamatia", "Rita Reang",
    "Bimal Debbarma", "Swapna Tripura", "Dipak Jamatia", "Mina Chakma",
    "Samir Debbarma", "Asha Tripura", "Pradip Reang", "Kabita Debbarma",
    "Tapan Jamatia", "Ujjwal Debbarma", "Prabir Chakma", "Anjali Reang",
    "Hemanta Debbarma", "Subrata Jamatia", "Kamala Tripura", "Nirmal Debbarma",
    "Gopal Reang", "Sanjay Murasing", "Deepa Jamatia", "Ratan Tripura",
    "Abhijit Tripura", "Amardweep Debbarma", "Ashis Kumar", "Amiya Kumar",
]

PLACES_EXPANDED = [
    # Existing + District HQs + Towns + Villages
    "Tripura", "Agartala", "Khumulwng", "Udaipur", "Dharmanagar",
    "Khowai", "Ambassa", "Belonia", "Sabroom", "Suryamaninagar",
    "Bishalgarh", "Santirbazar", "Jampui Hills", "Howrah", "Gomati",
    "New Delhi", "India", "Bangladesh", "Guwahati", "Kolkata",
    "Lunglei", "Kalyanpur", "Jirania", "Bodhjungnagar", "Melaghar",
    "Gorkhabasti", "Old Agartala", "Rudrasagar Lake", "Gomati River",
    # NEW: District headquarters and subdivisions from web
    "Bishramganj", "Kailashahar", "Kamalpur", "Teliamura",
    "Sonamura", "Mohanpur", "Amarpur", "Gandacherra",
    "Longtharai Valley", "Manu", "Kumarghat", "Panisagar",
    "Kadamtala", "Salema", "Boxanagar", "Hrishyamukh",
    # NEW: Rivers and geographic features
    "Haora River", "Manu River", "Feni River", "Khowai River",
    "Juri River", "Dhalai River", "Muhuri River", "Atharamura Hills",
    "Baramura Hills", "Deotamura Hills", "Longtharai Hills",
    # NEW: Neighboring areas
    "Assam", "Mizoram", "Manipur", "Meghalaya", "Chittagong",
    "Comilla", "Sylhet", "Dhaka", "Shillong", "Imphal", "Aizawl",
    # NEW: Indian cities
    "Delhi", "Mumbai", "Bengaluru", "Chennai", "Hyderabad",
]

ORGS_EXPANDED = [
    # Existing
    "Tripura Tribal Areas Autonomous District Council", "TTAADC",
    "Tipra Motha", "Tripura Hukbar Bedek", "Hukbar Bedek",
    "DD news", "Doordarshan", "Tripura University",
    "National Institute of Technology Agartala", "National Institute of Technology",
    "Indian Institute of Information Technology", "Maharaja Bir Bikram University",
    "MBB University", "Maharaja Bir Bikram College", "Tripura Medical College",
    "Twipra Students Federation", "TSF", "Kokborok Sahitya Sabha",
    "Kokborok Tei Hukumu Mission", "Tripura Legislative Assembly",
    "Government of Tripura", "Ministry of Education", "Press Trust of India",
    "Rubber Board of India", "Tripura State Museum", "Tata Technologies",
    "State Bank of India", "ONGC Tripura",
    # NEW: From web research
    "Agartala Government Medical College", "AGMC",
    "Govind Ballabh Pant Hospital", "GBP Hospital",
    "Tripura Santiniketan Medical College",
    "ICFAI University Tripura", "Techno India University",
    "National Law University Tripura", "Tripura Institute of Technology",
    "NIELIT Agartala", "SIPARD Tripura",
    "Women's Polytechnic Hapania", "Ramthakur College",
    "Women's College Agartala", "Iswar Chandra Vidyasagar College",
    "Holy Cross College", "Bir Bikram Memorial College",
    "National Forensic Sciences University",
    "Dhamma Dipa International Buddhist University",
    # NEW: Government departments and bodies
    "Tripura Police", "Tripura High Court", "Tripura Public Service Commission",
    "Tripura Gramin Bank", "Tripura State Rifles", "BSF Tripura",
    "Central Reserve Police Force", "Tripura State Electricity Corporation",
    "Tripura Industrial Development Corporation",
    "Tripura Forest Department", "Tripura Tourism Department",
    # NEW: Political parties
    "BJP Tripura", "CPI-M Tripura", "Congress Tripura",
    "IPFT", "Indigenous People's Front of Tripura",
    # NEW: Media
    "All India Radio Agartala", "Tripura Times", "Tripurainfo",
]

EVENTS_EXPANDED = [
    # Existing
    "Buisu", "Kharchi Puja", "Kokborok Day", "Garia Puja", "Ker Puja",
    "Neermahal Water Festival", "Durga Puja", "Diwali", "Christmas",
    # NEW: From web
    "Republic Day", "Independence Day", "Holi", "Eid",
    "Tripura Foundation Day", "Tring Festival",
    "Wangala Festival", "Sangrai Festival",
    "January 19", "Makar Sankranti",
    "Orange Festival", "Pineapple Festival",
    "World Tribal Day", "International Mother Language Day",
]

FACILITIES = [
    "Ujjayanta Palace", "Neermahal", "Maharaja Bir Bikram Airport",
    "MBB Airport", "Maharaja Bir Bikram Stadium", "Agartala Railway Station",
    "Tripura Sundari Temple", "Chaturdash Devata Bari",
    # NEW
    "Unakoti Rock Carvings", "Sepahijala Wildlife Sanctuary",
    "Dumbur Lake", "Jampui Orange Garden", "Heritage Park Agartala",
    "Fourteen Gods Temple", "Kamalasagar Kali Temple",
    "Pilak Archaeological Site", "Neermahal Water Palace",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Kokborok case suffixes
SUFFIXES = ["-o", "-ni", "-no", "-ao", "-bai", "-rok", "-yo", ""]

# Kokborok functional words for natural sentence construction
KOKBOROK_VERBS = [
    "phaikha", "khorok", "snammani", "tongo", "sanlaikha", "khwlaikha",
    "romjak kha", "wngkha", "chengmani", "rwlaikha", "salanai",
    "phai di", "phai mangnai", "kaham twi", "kokthwi kha",
    "chwlai kha", "gwrwng kha", "sainani", "thangwi kha", "rwkha",
    "kha", "nai", "ong kha", "chwng kha", "tangwi", "lanai",
]

KOKBOROK_CONNECTORS = [
    "aru", "tabuk", "bisi-o", "kok", "hotor", "kwrwi",
    "logot", "samung", "ni upor", "ni", "bai", "no",
]


def tag_tokens(text: str, label: str) -> list[tuple[str, str]]:
    """Convert a text string into BIO-tagged token tuples."""
    tokens = text.split()
    return [(tokens[i], f"{'B' if i == 0 else 'I'}-{label}") for i in range(len(tokens))]


def tag_with_suffix(text: str, label: str, suffix: str = "") -> list[tuple[str, str]]:
    """Tag tokens with BIO and optionally attach suffix to last token."""
    tokens = text.split()
    result = []
    for i, tok in enumerate(tokens):
        prefix = "B-" if i == 0 else "I-"
        if i == len(tokens) - 1 and suffix:
            result.append((f"{tok}{suffix}", f"{prefix}{label}"))
        else:
            result.append((tok, f"{prefix}{label}"))
    return result


def o(*words: str) -> list[tuple[str, str]]:
    """Create O-tagged token tuples."""
    return [(w, "O") for w in words]


def generate_templates(count: int = 5000) -> list[list[tuple[str, str]]]:
    """Generate diverse Kokborok NER training sentences from 40 templates."""
    sentences: list[list[tuple[str, str]]] = []

    for idx in range(count):
        t = idx % 40  # 40 diverse templates
        per = random.choice(PERSONS_EXPANDED)
        per2 = random.choice([p for p in PERSONS_EXPANDED if p != per])
        loc = random.choice(PLACES_EXPANDED)
        loc2 = random.choice([l for l in PLACES_EXPANDED if l != loc])
        org = random.choice(ORGS_EXPANDED)
        org2 = random.choice([o2 for o2 in ORGS_EXPANDED if o2 != org])
        event = random.choice(EVENTS_EXPANDED)
        fac = random.choice(FACILITIES)
        month = random.choice(MONTHS)
        year = str(random.choice(range(1979, 2027)))
        day = str(random.randint(1, 28))
        suffix = random.choice(SUFFIXES)
        verb = random.choice(KOKBOROK_VERBS)

        tokens: list[tuple[str, str]] = []

        # ===================== PER-HEAVY TEMPLATES (0-12) =====================
        if t == 0:
            # [PER] wngkha [ORG]-ni chairman .
            tokens = tag_tokens(per, "PER") + o("wngkha") + tag_with_suffix(org, "ORG", "-ni") + o("chairman", ".")
        elif t == 1:
            # [PER] aru [PER2] [LOC]-o meeting khorok .
            tokens = tag_tokens(per, "PER") + o("aru") + tag_tokens(per2, "PER") + tag_with_suffix(loc, "LOC", "-o") + o("meeting", "khorok", ".")
        elif t == 2:
            # Minister [PER] [ORG]-ni function-o phaikha .
            tokens = o("Minister") + tag_tokens(per, "PER") + tag_with_suffix(org, "ORG", "-ni") + o("function-o", "phaikha", ".")
        elif t == 3:
            # [PER] , [PER2] aru [PER3] [EVENT]-o sabai tongo .
            per3 = random.choice([p for p in PERSONS_EXPANDED if p != per and p != per2])
            tokens = tag_tokens(per, "PER") + o(",") + tag_tokens(per2, "PER") + o("aru") + tag_tokens(per3, "PER") + tag_with_suffix(event, "MISC", "-o") + o("sabai", "tongo", ".")
        elif t == 4:
            # [PER] wngkha [LOC]-ni MLA .
            tokens = tag_tokens(per, "PER") + o("wngkha") + tag_with_suffix(loc, "LOC", "-ni") + o("MLA", ".")
        elif t == 5:
            # Doctor [PER] [ORG]-o treatment sanlaikha .
            tokens = o("Doctor") + tag_tokens(per, "PER") + tag_with_suffix(org, "ORG", "-o") + o("treatment", "sanlaikha", ".")
        elif t == 6:
            # Teacher [PER] [ORG]-o padhai sanlaikha .
            tokens = o("Teacher") + tag_tokens(per, "PER") + tag_with_suffix(org, "ORG", "-o") + o("padhai", "sanlaikha", ".")
        elif t == 7:
            # Athlete [PER] [LOC]-ni sports competition-o khwlaikha .
            tokens = o("Athlete") + tag_tokens(per, "PER") + tag_with_suffix(loc, "LOC", "-ni") + o("sports", "competition-o", "khwlaikha", ".")
        elif t == 8:
            # [PER]-ni family [LOC]-o tongo .
            tokens = tag_with_suffix(per, "PER", "-ni") + o("family") + tag_with_suffix(loc, "LOC", "-o") + o("tongo", ".")
        elif t == 9:
            # [PER] [LOC] aru [LOC2]-o phaikha .
            tokens = tag_tokens(per, "PER") + tag_tokens(loc, "LOC") + o("aru") + tag_with_suffix(loc2, "LOC", "-o") + o("phaikha", ".")
        elif t == 10:
            # Chief Minister [PER] [LOC]-o project launch khorok .
            tokens = o("Chief", "Minister") + tag_tokens(per, "PER") + tag_with_suffix(loc, "LOC", "-o") + o("project", "launch", "khorok", ".")
        elif t == 11:
            # [PER] wngkha [ORG]-ni founder , [LOC]-ni borokrog .
            tokens = tag_tokens(per, "PER") + o("wngkha") + tag_with_suffix(org, "ORG", "-ni") + o("founder", ",") + tag_with_suffix(loc, "LOC", "-ni") + o("borokrog", ".")
        elif t == 12:
            # Writer [PER] Kokborok literature-ni kitab brw kha .
            tokens = o("Writer") + tag_tokens(per, "PER") + o("Kokborok", "literature-ni", "kitab", "brw", "kha", ".")

        # ===================== ORG-HEAVY TEMPLATES (13-22) =====================
        elif t == 13:
            # [ORG] aru [ORG2] [LOC]-o joint office tongo .
            tokens = tag_tokens(org, "ORG") + o("aru") + tag_tokens(org2, "ORG") + tag_with_suffix(loc, "LOC", "-o") + o("joint", "office", "tongo", ".")
        elif t == 14:
            # [ORG]-ni employee rok [LOC]-o protest khorok .
            tokens = tag_with_suffix(org, "ORG", "-ni") + o("employee", "rok") + tag_with_suffix(loc, "LOC", "-o") + o("protest", "khorok", ".")
        elif t == 15:
            # [ORG] [YEAR]-o established khorok [LOC]-o .
            tokens = tag_tokens(org, "ORG") + o(year + "-o", "established", "khorok") + tag_with_suffix(loc, "LOC", "-o") + o(".")
        elif t == 16:
            # [PER] [ORG]-ni secretary wngkha .
            tokens = tag_tokens(per, "PER") + tag_with_suffix(org, "ORG", "-ni") + o("secretary", "wngkha", ".")
        elif t == 17:
            # [ORG]-ni meeting [LOC]-o [MONTH] [DAY] chengmani .
            tokens = tag_with_suffix(org, "ORG", "-ni") + o("meeting") + tag_with_suffix(loc, "LOC", "-o") + tag_tokens(month, "MISC") + [(day, "I-MISC")] + o("chengmani", ".")
        elif t == 18:
            # [ORG] [LOC] aru [LOC2]-o branch tongo .
            tokens = tag_tokens(org, "ORG") + tag_tokens(loc, "LOC") + o("aru") + tag_with_suffix(loc2, "LOC", "-o") + o("branch", "tongo", ".")
        elif t == 19:
            # [PER] [ORG]-o award sonan [EVENT]-o .
            tokens = tag_tokens(per, "PER") + tag_with_suffix(org, "ORG", "-o") + o("award", "sonan") + tag_with_suffix(event, "MISC", "-o") + o(".")
        elif t == 20:
            # [ORG]-ni principal [PER] [LOC]-o tongo .
            tokens = tag_with_suffix(org, "ORG", "-ni") + o("principal") + tag_tokens(per, "PER") + tag_with_suffix(loc, "LOC", "-o") + o("tongo", ".")
        elif t == 21:
            # [ORG] [LOC]-o new campus rwlaikha .
            tokens = tag_tokens(org, "ORG") + tag_with_suffix(loc, "LOC", "-o") + o("new", "campus", "rwlaikha", ".")
        elif t == 22:
            # [ORG]-ni student rok [EVENT] chengmani khorok .
            tokens = tag_with_suffix(org, "ORG", "-ni") + o("student", "rok") + tag_tokens(event, "MISC") + o("chengmani", "khorok", ".")

        # ===================== LOC-HEAVY TEMPLATES (23-29) =====================
        elif t == 23:
            # [LOC]-ni borokrog [LOC2]-o phaikha .
            tokens = tag_with_suffix(loc, "LOC", "-ni") + o("borokrog") + tag_with_suffix(loc2, "LOC", "-o") + o("phaikha", ".")
        elif t == 24:
            # [LOC] wngkha [LOC2]-ni pande .
            tokens = tag_tokens(loc, "LOC") + o("wngkha") + tag_with_suffix(loc2, "LOC", "-ni") + o("pande", ".")
        elif t == 25:
            # [LOC] aru [LOC2]-ni majao road tongo .
            tokens = tag_tokens(loc, "LOC") + o("aru") + tag_with_suffix(loc2, "LOC", "-ni") + o("majao", "road", "tongo", ".")
        elif t == 26:
            # [PER] [LOC]-ni [FAC] snammani .
            tokens = tag_tokens(per, "PER") + tag_with_suffix(loc, "LOC", "-ni") + tag_tokens(fac, "LOC") + o("snammani", ".")
        elif t == 27:
            # [LOC]-o heavy rain twilom kha .
            tokens = tag_with_suffix(loc, "LOC", "-o") + o("heavy", "rain", "twilom", "kha", ".")
        elif t == 28:
            # [LOC] aru [LOC2] wngkha [ORG]-ni district .
            tokens = tag_tokens(loc, "LOC") + o("aru") + tag_tokens(loc2, "LOC") + o("wngkha") + tag_with_suffix(org, "ORG", "-ni") + o("district", ".")
        elif t == 29:
            # Train [LOC] tok [LOC2]-o chalaiwi .
            tokens = o("Train") + tag_tokens(loc, "LOC") + o("tok") + tag_with_suffix(loc2, "LOC", "-o") + o("chalaiwi", ".")

        # ===================== MISC + MIXED TEMPLATES (30-39) =====================
        elif t == 30:
            # [EVENT] romwi [MONTH] [DAY] , [YEAR] bisi-o chengmani .
            tokens = tag_tokens(event, "MISC") + o("romwi") + tag_tokens(month, "MISC") + [(day, "I-MISC")] + o(",") + [(year, "B-MISC")] + o("bisi-o", "chengmani", ".")
        elif t == 31:
            # [LOC]-ni borokrog [EVENT] romwi rwng kthar khwlai-o .
            tokens = tag_with_suffix(loc, "LOC", "-ni") + o("borokrog") + tag_tokens(event, "MISC") + o("romwi", "rwng", "kthar", "khwlai-o", ".")
        elif t == 32:
            # [MONTH] [DAY] [YEAR]-ot [PER] [LOC]-o phaikha .
            tokens = tag_tokens(month, "MISC") + [(day, "I-MISC")] + [(year + "-ot", "I-MISC")] + tag_tokens(per, "PER") + tag_with_suffix(loc, "LOC", "-o") + o("phaikha", ".")
        elif t == 33:
            # [PER] [ORG]-ni [EVENT]-o speech sanlaikha .
            tokens = tag_tokens(per, "PER") + tag_with_suffix(org, "ORG", "-ni") + tag_with_suffix(event, "MISC", "-o") + o("speech", "sanlaikha", ".")
        elif t == 34:
            # [ORG] [LOC]-o [EVENT] chengmani khorok [YEAR]-o .
            tokens = tag_tokens(org, "ORG") + tag_with_suffix(loc, "LOC", "-o") + tag_tokens(event, "MISC") + o("chengmani", "khorok") + [(year + "-o", "B-MISC")] + o(".")
        elif t == 35:
            # [PER] , [ORG]-ni padhai , [LOC]-o kothoma sanlaikha .
            tokens = tag_tokens(per, "PER") + o(",") + tag_with_suffix(org, "ORG", "-ni") + o("padhai", ",") + tag_with_suffix(loc, "LOC", "-o") + o("kothoma", "sanlaikha", ".")
        elif t == 36:
            # [PER] [ORG]-o [LOC]-ni research project-ot khorok .
            tokens = tag_tokens(per, "PER") + tag_with_suffix(org, "ORG", "-o") + tag_with_suffix(loc, "LOC", "-ni") + o("research", "project-ot", "khorok", ".")
        elif t == 37:
            # [LOC]-ni [ORG] [PER]-ni padhai sanlaikha .
            tokens = tag_with_suffix(loc, "LOC", "-ni") + tag_tokens(org, "ORG") + tag_with_suffix(per, "PER", "-ni") + o("padhai", "sanlaikha", ".")
        elif t == 38:
            # Kokborok language [EVENT]-o [LOC]-ni [ORG] discussion khorok .
            tokens = o("Kokborok", "language") + tag_with_suffix(event, "MISC", "-o") + tag_with_suffix(loc, "LOC", "-ni") + tag_tokens(org, "ORG") + o("discussion", "khorok", ".")
        elif t == 39:
            # [DAY] [MONTH] [YEAR]-ot [LOC]-ao [ORG]-ao [PER] logot meeting khorok .
            tokens = [(day, "B-MISC"), (month, "I-MISC"), (year + "-ot", "I-MISC")] + tag_with_suffix(loc, "LOC", "-ao") + tag_with_suffix(org, "ORG", "-ao") + tag_tokens(per, "PER") + o("logot", "meeting", "khorok", ".")

        if tokens:
            sentences.append(tokens)

    return sentences


def entity_substitution_augment(
    sentences: list[list[tuple[str, str]]],
    n_copies: int = 8,
) -> list[list[tuple[str, str]]]:
    """
    For each sentence containing entities, create n_copies with randomly
    substituted entities of the same type + random Kokborok suffix variations.
    """
    augmented: list[list[tuple[str, str]]] = []

    entity_pools = {
        "PER": PERSONS_EXPANDED,
        "LOC": PLACES_EXPANDED + [f.split()[0] for f in FACILITIES],
        "ORG": ORGS_EXPANDED,
        "MISC": EVENTS_EXPANDED + MONTHS,
    }

    for sent in sentences:
        # Check if sentence has entities
        has_entity = any(tag != "O" for _, tag in sent)
        if not has_entity:
            continue

        for _ in range(n_copies):
            new_sent: list[tuple[str, str]] = []
            i = 0
            while i < len(sent):
                tok, tag = sent[i]
                if tag.startswith("B-"):
                    etype = tag[2:]
                    # Collect the full entity span
                    span_tokens = [tok]
                    j = i + 1
                    while j < len(sent) and sent[j][1] == f"I-{etype}":
                        span_tokens.append(sent[j][0])
                        j += 1

                    # Choose a random replacement entity
                    pool = entity_pools.get(etype, [])
                    if pool:
                        new_entity = random.choice(pool)
                        new_toks = new_entity.split()

                        # Optionally attach suffix to last token
                        suffix = random.choice(SUFFIXES)
                        # Check if original last token had a suffix
                        orig_last = span_tokens[-1]
                        for s in ["-o", "-ni", "-no", "-ao", "-bai", "-rok", "-yo", "-ot"]:
                            if orig_last.endswith(s):
                                suffix = s
                                break

                        for k, nt in enumerate(new_toks):
                            bio = "B-" if k == 0 else "I-"
                            if k == len(new_toks) - 1 and suffix:
                                new_sent.append((f"{nt}{suffix}", f"{bio}{etype}"))
                            else:
                                new_sent.append((nt, f"{bio}{etype}"))
                    else:
                        # Keep original
                        for k, st in enumerate(span_tokens):
                            bio = "B-" if k == 0 else "I-"
                            new_sent.append((st, f"{bio}{etype}"))

                    i = j
                else:
                    new_sent.append((tok, tag))
                    i += 1

            augmented.append(new_sent)

    return augmented


def extract_excel_sentences() -> list[str]:
    """Extract Kokborok sentences from all available Excel datasets."""
    try:
        import openpyxl
    except ImportError:
        print("[WARN] openpyxl not installed, skipping Excel extraction")
        return []

    base = PROJECT_ROOT / "data" / "raw" / "kokborok"
    if not base.exists():
        return []

    all_sentences: list[str] = []

    for fn in os.listdir(base):
        if not fn.endswith(".xlsx"):
            continue
        fp = base / fn
        try:
            wb = openpyxl.load_workbook(fp, data_only=True)
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                try:
                    _ = ws.max_row
                except:
                    continue

                for row in range(1, ws.max_row + 1):
                    for col in range(1, ws.max_column + 1):
                        val = ws.cell(row, col).value
                        if val and isinstance(val, str):
                            val = val.strip()
                            # Look for Kokborok text (contains Kokborok words or markers)
                            if len(val) > 10 and any(c.isalpha() for c in val):
                                # Skip English-only headers and labels
                                if val in ("English Sentences", "Kokborok Sentences", "None",
                                           "Sentiment Analysis Tagging", "English Questions",
                                           "Kokborok Answers", "ENGLISH SENTENCES",
                                           "KOKBOROK TRANSLATIONS", "Kokborok Sarcasm sentences",
                                           "Sarcasm Tagging"):
                                    continue
                                all_sentences.append(val)
        except Exception as e:
            print(f"[WARN] Could not read {fn}: {e}")

    return all_sentences


def auto_tag_sentence(
    sentence: str,
    per_gaz: set[str],
    loc_gaz: set[str],
    org_gaz: set[str],
    misc_gaz: set[str],
) -> list[tuple[str, str]] | None:
    """
    Auto-tag a sentence using gazetteer matching.
    Returns None if no entities found (skip for training efficiency).
    """
    from preprocessing.kokborok_tokenizer import KokborokTokenizer
    tok = KokborokTokenizer()
    tokens = tok.tokenize_words(sentence)
    if not tokens or len(tokens) < 3:
        return None

    tags = ["O"] * len(tokens)
    found_entity = False

    # Multi-token and single-token gazetteer matching
    for gaz, label in [(per_gaz, "PER"), (loc_gaz, "LOC"), (org_gaz, "ORG"), (misc_gaz, "MISC")]:
        for entry in gaz:
            entry_tokens = entry.split()
            entry_len = len(entry_tokens)

            for i in range(len(tokens) - entry_len + 1):
                # Check for exact match (also check with suffix stripped)
                match = True
                for j in range(entry_len):
                    clean_tok = tokens[i + j]
                    # Strip Kokborok suffixes for matching
                    for suf in ["-o", "-ni", "-no", "-ao", "-bai", "-rok", "-yo", "-ot"]:
                        if clean_tok.endswith(suf) and len(clean_tok) > len(suf) + 1:
                            clean_tok = clean_tok[:-len(suf)]
                            break
                    if clean_tok.lower() != entry_tokens[j].lower():
                        match = False
                        break

                if match and all(tags[i + k] == "O" for k in range(entry_len)):
                    tags[i] = f"B-{label}"
                    for k in range(1, entry_len):
                        tags[i + k] = f"I-{label}"
                    found_entity = True

    if not found_entity:
        return None

    return list(zip(tokens, tags))


def load_conll(path: Path) -> list[list[tuple[str, str]]]:
    """Load CoNLL format file."""
    sentences = []
    curr: list[tuple[str, str]] = []
    if not path.is_file():
        return sentences
    with open(path, "r", encoding="utf-8") as f:
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
    return sentences


def save_conll(sentences: list[list[tuple[str, str]]], path: Path) -> None:
    """Save sentences in CoNLL format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for tok, tag in sent:
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")


def count_entities(sentences: list[list[tuple[str, str]]]) -> dict[str, int]:
    """Count B- tags per entity type."""
    counts: dict[str, int] = {"PER": 0, "LOC": 0, "ORG": 0, "MISC": 0}
    for sent in sentences:
        for _, tag in sent:
            if tag.startswith("B-"):
                etype = tag[2:]
                if etype in counts:
                    counts[etype] += 1
    return counts


def build_training_data() -> Path:
    """Build the comprehensive Kokborok training dataset."""
    random.seed(42)
    print("=" * 70)
    print("KOKBOROK NER IMPROVEMENT PIPELINE — DATA GENERATION")
    print("=" * 70)

    all_sentences: list[list[tuple[str, str]]] = []

    # 1. Load original gold training data
    gold_path = PROJECT_ROOT / "data" / "gold" / "kokborok" / "train.conll"
    gold_sents = load_conll(gold_path)
    all_sentences.extend(gold_sents)
    print(f"\n[1] Gold train: {len(gold_sents)} sentences")
    gold_counts = count_entities(gold_sents)
    print(f"    Entities: {gold_counts}")

    # 2. Load silver data
    silver_path = PROJECT_ROOT / "data" / "silver" / "kokborok" / "train.conll"
    silver_sents = load_conll(silver_path)
    all_sentences.extend(silver_sents)
    print(f"\n[2] Silver train: {len(silver_sents)} sentences")
    silver_counts = count_entities(silver_sents)
    print(f"    Entities: {silver_counts}")

    # 3. Extract and auto-tag Excel data
    print(f"\n[3] Extracting from Excel datasets...")
    excel_sentences = extract_excel_sentences()
    print(f"    Found {len(excel_sentences)} raw sentences from Excel")

    # Build gazetteer sets for matching
    per_gaz = set(PERSONS_EXPANDED)
    loc_gaz = set(PLACES_EXPANDED)
    org_gaz = set(ORGS_EXPANDED)
    misc_gaz = set(EVENTS_EXPANDED)

    excel_tagged = 0
    for sent_text in excel_sentences:
        tagged = auto_tag_sentence(sent_text, per_gaz, loc_gaz, org_gaz, misc_gaz)
        if tagged:
            all_sentences.append(tagged)
            excel_tagged += 1
    print(f"    Auto-tagged: {excel_tagged} sentences with entities")

    # 4. Generate 5,000 diverse synthetic sentences from 40 templates
    print(f"\n[4] Generating 5,000 synthetic sentences from 40 templates...")
    synthetic = generate_templates(5000)
    all_sentences.extend(synthetic)
    synth_counts = count_entities(synthetic)
    print(f"    Generated: {len(synthetic)} sentences")
    print(f"    Entities: {synth_counts}")

    # 5. Entity substitution augmentation (×8 copies of entity-containing sentences)
    print(f"\n[5] Entity substitution augmentation (×8 copies)...")
    # Only augment gold + excel tagged sentences (not synthetic — they're already diverse)
    entity_sents = [s for s in gold_sents + silver_sents if any(t != "O" for _, t in s)]
    augmented = entity_substitution_augment(entity_sents, n_copies=8)
    all_sentences.extend(augmented)
    aug_counts = count_entities(augmented)
    print(f"    Augmented: {len(augmented)} sentences")
    print(f"    Entities: {aug_counts}")

    # 6. Also augment a subset of the synthetic sentences for extra diversity
    print(f"\n[6] Augmenting synthetic subset...")
    synth_with_entities = [s for s in synthetic if any(t != "O" for _, t in s)]
    synth_sample = random.sample(synth_with_entities, min(500, len(synth_with_entities)))
    synth_aug = entity_substitution_augment(synth_sample, n_copies=4)
    all_sentences.extend(synth_aug)
    saug_counts = count_entities(synth_aug)
    print(f"    Augmented: {len(synth_aug)} synthetic sentences")
    print(f"    Entities: {saug_counts}")

    # Final stats
    total_counts = count_entities(all_sentences)
    print(f"\n{'=' * 70}")
    print(f"FINAL DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total sentences: {len(all_sentences):,}")
    print(f"Entity counts:")
    for etype, cnt in sorted(total_counts.items()):
        print(f"  B-{etype}: {cnt:,}")
    total_ents = sum(total_counts.values())
    total_toks = sum(len(s) for s in all_sentences)
    print(f"  Total entity instances: {total_ents:,}")
    print(f"  Total tokens: {total_toks:,}")
    print(f"  Entity density: {total_ents / total_toks * 100:.1f}%")

    # Save
    out_path = PROJECT_ROOT / "data" / "processed" / "kokborok_improved_train.conll"
    save_conll(all_sentences, out_path)
    print(f"\n[SAVED] {out_path.relative_to(PROJECT_ROOT)} ({out_path.stat().st_size:,} bytes)")

    return out_path


# ============================================================================
# PHASE 2: RETRAIN MODEL
# ============================================================================

def retrain_model(train_data_path: Path) -> Path:
    """Retrain Kokborok with Option-C protected backbone."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from model.ner_model_v4_multilingual import NERMultilingualModel
    from training.multilingual_dataset import MultilingualNERDataset, collate_fn_multilingual
    from evaluation.seqeval_metrics import compute_seqeval_metrics
    from training.train_charcnn import ID_TO_TAG, TAG_TO_ID
    from utils.seed import set_global_seed

    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}")
    print(f"KOKBOROK MODEL RETRAINING — Option C Protected Backbone")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"{'=' * 70}")

    # Load vocabs
    bundle_dir = PROJECT_ROOT / "deployment" / "bundles" / "kokborok" / "v1"
    with open(bundle_dir / "word_vocab.json", "r", encoding="utf-8") as f:
        word_to_id = json.load(f).get("token_to_id", {})
    with open(bundle_dir / "char_vocab.json", "r", encoding="utf-8") as f:
        char_to_id = json.load(f).get("token_to_id", {})

    print(f"Vocab: {len(word_to_id):,} words, {len(char_to_id):,} chars")

    # Initialize model
    model = NERMultilingualModel(
        word_vocab_size=len(word_to_id),
        char_vocab_size=len(char_to_id),
        num_tags=9,
        num_languages=4,
        crf_config="per_language",
        hidden_dim=128,
    ).to(device)

    # Load current best checkpoint
    ckpt_path = bundle_dir / "model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    print(f"Loaded checkpoint from {ckpt_path}")

    # Option C: Freeze backbone, train only Kokborok-specific components
    # We'll unfreeze everything but protect existing languages' heads
    frozen_count = 0
    unfrozen_count = 0
    for name, param in model.named_parameters():
        # Freeze other languages' CRF and FC heads (indices 0, 1, 2)
        if any(f"fc_heads.{i}" in name or f"crf_heads.{i}" in name for i in [0, 1, 2]):
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = True
            unfrozen_count += param.numel()

    print(f"Frozen parameters (other langs): {frozen_count:,}")
    print(f"Trainable parameters: {unfrozen_count:,}")

    # Datasets
    dev_path = PROJECT_ROOT / "data" / "gold" / "kokborok" / "dev.conll"

    train_ds = MultilingualNERDataset(
        lang_conll_paths={"kokborok": [train_data_path]},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )
    dev_ds = MultilingualNERDataset(
        lang_conll_paths={"kokborok": [dev_path]},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn_multilingual)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)

    print(f"Train: {len(train_ds):,} samples, Dev: {len(dev_ds)} samples")

    # Training setup
    max_epochs = 20
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=5e-5)

    best_dev_f1 = -1.0
    best_state = None
    best_metrics = None
    patience = 5
    no_improve = 0

    print(f"\nTraining for {max_epochs} epochs...\n")
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            w = batch["word_ids"].to(device)
            c = batch["char_ids"].to(device)
            l_ids = batch["lang_ids"].to(device)
            t = batch["tags"].to(device)
            m = batch["mask"].to(device)

            loss = model(w, c, lang_ids=l_ids, tags=t, mask=m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0
            )
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Dev evaluation
        model.eval()
        all_gold, all_pred = [], []
        with torch.no_grad():
            for batch in dev_loader:
                w = batch["word_ids"].to(device)
                c = batch["char_ids"].to(device)
                l_ids = batch["lang_ids"].to(device)
                t = batch["tags"].to(device)
                m = batch["mask"].to(device)
                seq_lengths = batch["seq_lengths"]

                paths = model(w, c, lang_ids=l_ids, mask=m)
                for i, slen in enumerate(seq_lengths):
                    all_gold.append([ID_TO_TAG[tid.item()] for tid in t[i, :slen]])
                    all_pred.append([ID_TO_TAG[tid] for tid in paths[i][:slen]])

        dev_metrics = compute_seqeval_metrics(all_gold, all_pred)
        dev_f1 = dev_metrics["overall"]["entity_f1"] * 100.0
        dev_p = dev_metrics["overall"]["entity_precision"] * 100.0
        dev_r = dev_metrics["overall"]["entity_recall"] * 100.0

        # Per-type F1
        type_str = " | ".join(
            f"{t}: {dev_metrics['by_type'][t]['f1']*100:.1f}%"
            for t in ["PER", "LOC", "ORG", "MISC"]
        )

        avg_loss = total_loss / max(len(train_loader), 1)
        lr = scheduler.get_last_lr()[0]
        marker = ""
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dev_metrics
            marker = " ★ BEST"
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"  Epoch {epoch:2d}/{max_epochs} | Loss: {avg_loss:.4f} (lr: {lr:.1e}) | "
            f"F1: {dev_f1:5.2f}% (P: {dev_p:5.2f}%, R: {dev_r:5.2f}%) | {type_str}{marker}"
        )

        if no_improve >= patience and epoch >= 10:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Save best checkpoint
    save_path = PROJECT_ROOT / "checkpoints" / "kokborok_improved_v2.pt"
    torch.save({"model_state_dict": best_state, "dev_metrics": best_metrics}, save_path)
    print(f"\n[SAVED] Best checkpoint: {save_path.relative_to(PROJECT_ROOT)}")
    print(f"Best Dev F1: {best_dev_f1:.2f}%")

    # Per-type breakdown
    if best_metrics:
        print(f"\nPer-Type Dev Results:")
        for t in ["PER", "LOC", "ORG", "MISC"]:
            m = best_metrics["by_type"][t]
            print(f"  {t}: P={m['precision']*100:.1f}%, R={m['recall']*100:.1f}%, F1={m['f1']*100:.1f}%")

    return save_path


# ============================================================================
# PHASE 3: EVALUATE ON LOCKED GOLD-TEST
# ============================================================================

def evaluate_on_gold_test(ckpt_path: Path) -> dict[str, Any]:
    """Run final evaluation on locked gold-test split."""
    import torch
    from torch.utils.data import DataLoader

    from model.ner_model_v4_multilingual import NERMultilingualModel
    from training.multilingual_dataset import MultilingualNERDataset, collate_fn_multilingual
    from evaluation.seqeval_metrics import compute_seqeval_metrics
    from training.train_charcnn import ID_TO_TAG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}")
    print(f"GOLD-TEST EVALUATION")
    print(f"{'=' * 70}")

    # Load vocabs
    bundle_dir = PROJECT_ROOT / "deployment" / "bundles" / "kokborok" / "v1"
    with open(bundle_dir / "word_vocab.json", "r", encoding="utf-8") as f:
        word_to_id = json.load(f).get("token_to_id", {})
    with open(bundle_dir / "char_vocab.json", "r", encoding="utf-8") as f:
        char_to_id = json.load(f).get("token_to_id", {})

    # Load model
    model = NERMultilingualModel(
        word_vocab_size=len(word_to_id),
        char_vocab_size=len(char_to_id),
        num_tags=9,
        num_languages=4,
        crf_config="per_language",
        hidden_dim=128,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # Load test data
    test_path = PROJECT_ROOT / "data" / "gold" / "kokborok" / "test.conll"
    test_ds = MultilingualNERDataset(
        lang_conll_paths={"kokborok": [test_path]},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)

    all_gold, all_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            w = batch["word_ids"].to(device)
            c = batch["char_ids"].to(device)
            l_ids = batch["lang_ids"].to(device)
            t = batch["tags"].to(device)
            m = batch["mask"].to(device)
            seq_lengths = batch["seq_lengths"]

            paths = model(w, c, lang_ids=l_ids, mask=m)
            for i, slen in enumerate(seq_lengths):
                all_gold.append([ID_TO_TAG[tid.item()] for tid in t[i, :slen]])
                all_pred.append([ID_TO_TAG[tid] for tid in paths[i][:slen]])

    metrics = compute_seqeval_metrics(all_gold, all_pred)

    print(f"\n{'─' * 50}")
    print(f"KOKBOROK GOLD-TEST RESULTS (Improved Model)")
    print(f"{'─' * 50}")
    print(f"Overall Entity F1:   {metrics['overall']['entity_f1']*100:.2f}%")
    print(f"Overall Precision:   {metrics['overall']['entity_precision']*100:.2f}%")
    print(f"Overall Recall:      {metrics['overall']['entity_recall']*100:.2f}%")
    print(f"Token Accuracy:      {metrics['overall']['token_accuracy']*100:.2f}%")

    print(f"\nPer-Entity-Type Breakdown:")
    print(f"{'Entity':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"{'─'*55}")
    for t in ["PER", "LOC", "ORG", "MISC"]:
        m = metrics["by_type"][t]
        print(f"{t:<8} {m['precision']*100:>9.2f}% {m['recall']*100:>9.2f}% {m['f1']*100:>9.2f}% {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")

    return metrics


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("=  KOKBOROK NER IMPROVEMENT PIPELINE - FULL EXECUTION")
    print("=" * 70)

    # Phase 1: Build expanded training data
    train_path = build_training_data()

    # Phase 2: Retrain model
    ckpt_path = retrain_model(train_path)

    # Phase 3: Evaluate on gold-test
    metrics = evaluate_on_gold_test(ckpt_path)

    print("\n" + "=" * 70)
    print("=  PIPELINE COMPLETE")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    main()
