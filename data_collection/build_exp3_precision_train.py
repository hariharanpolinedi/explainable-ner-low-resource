"""
Experiment 3: Precision-Recovery & Hard-Negative Augmented Kokborok Dataset.

Interventions:
1. Preserve Recall: Full coverage of all genuine Kokborok entities, clitic suffixes (-o, ni, -ni),
   and casing variants.
2. Hard Negatives for Clitic Disambiguation:
   - Common Kokborok nouns with locative '-o' and genitive 'ni' (e.g. Nok-o, Huk-o, Twi-o, Haste-o,
     Samung ni, Luku ni, Dophani, Borok-ni, Kaitor-ni) tagged strictly as 'O'.
3. Hard Negatives for Sentence-Initial Capitalized Verbs & Common Words:
   - Imperatives (Swnamdi, Khwlaidi, Phwlaidi, Naidi, Thangdi, Phaidi, Swrwngdi) tagged as 'O'.
   - Common concepts (Jwngjal, Gunda, Bank, Hospital, College, School, Meeting, Order, Rules) tagged as 'O'.
   - Technical/Loan words (Software, Sensor, Signal, Filter, Register, Social media, Internet) tagged as 'O'.
4. Strict Zero Leakage:
   - Uses independent tribal vocabulary and linguistic grammar patterns.
   - Never accesses dev.conll or test.conll sentences.
"""

import random
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path("c:/Users/harih/Desktop/NLP-MARK-3")
GOLD_TRAIN_FILE = PROJECT_ROOT / "data" / "gold" / "kokborok" / "train.conll"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "kokborok_exp3_precision_train.conll"

random.seed(42)

# 1. Base entities
ENTITIES = {
    "PER": [
        "Debbarma", "Tripura", "Jamatia", "Reang", "Noatia", "Chakma", "Kalai", 
        "Murasing", "Koloi", "Rupini", "Hrangkhawl", "Uchoi", "Galo", "Darlong",
        "Bir Bikram", "Radha Kishore", "Birendra Kishore", "Kirit Pradyot", "Manikya",
        "Bimal Debbarma", "Animesh Debbarma", "Bikash Tripura", "Binoy Debbarma",
        "Athlete Amardweep Debbarma", "Bijoy Kumar Hrangkhawl", "Biplab Kumar Deb",
        "Bipasha Tripura", "Asha Tripura", "Bidhan Tripura", "Parimal Debbarma", "Parimal",
        "Kanika", "kanika", "Rajiv Gandhi", "Indira Gandhi", "Jawaharlal Nehru",
        "Ram", "Rajiv", "Sanjay", "Jisu Khristo", "Jisu", "Mohan", "Rahul", "Sunil"
    ],
    "LOC": [
        "Agartala", "Khowai", "Udaipur", "Dharmanagar", "Ambassa", "Kailashahar", 
        "Belonia", "Santirbazar", "Teliamura", "Sabroom", "Bishalgarh", "Sonamura", 
        "Amarpur", "Kanchanpur", "Gandacherra", "Kumarghat", "Jampui", "Baramura", 
        "Atharamura", "Longthorai", "Deo", "Gomati", "Howrah", "Manu", "Dhalai", 
        "Muhuri", "Feni", "Radhapur", "Khumpui", "Jampui Hills", "Tripura",
        "Dashamighat", "Boxanagar", "Chabimura", "Battala", "Akhaura",
        "Barot", "India", "England", "Delhi", "Kolkata", "Assam", "Mizoram", "Manipur", "Guwahati",
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
        "DD news", "DD News", "Doordarshan", "Tripura board", "Tripura Board",
        "TIPRA Motha", "INPT", "IPFT", "BJP", "Congress", "CPI(M)", "CPIM",
        "Tripura Gramin Bank", "TGB", "TRTC", "TSECL", "TTDC", "ONGC Tripura", "ONGC", 
        "State Bank of India", "SBI", "TFTC", "THFC"
    ],
    "MISC": [
        "Buisu", "Kharchi Puja", "Kokborok Day", "Garia Puja", "Ker Puja",
        "Wangala", "Hojagiri", "Durga Puja", "Tripuri", "Hojagiri Dance", "Christianity", "Christmas"
    ]
}

# 2. Hard Negative Vocabulary (Strictly 'O')
COMMON_CLITIC_NOUNS = [
    ("Nok-o", "in the house"), ("Huk-o", "in the jhum field"), ("Twi-o", "in the water"),
    ("Haste-o", "in the state"), ("Luku-o", "among the people"), ("Khorok-o", "on the head"),
    ("Samung-o", "at work"), ("Dopha-o", "in the clan"), ("Phola-o", "in the spirit"),
    ("Lam-o", "on the road"), ("Nokha-o", "in the sky"), ("Dophani", "of the clan"),
    ("Lukuni", "of the people"), ("Borokni", "of the human/Borok"), ("Samungni", "of the work"),
    ("Hukni", "of the field"), ("Twini", "of the water"), ("Haste-ni", "of the state"),
    ("Kokborok-ni", "of the Kokborok language"), ("Bwsani", "of the child"), ("Kaitorni", "of God"),
    ("Buma-ni", "of the mother"), ("Bpha-ni", "of the father"), ("Bini", "his/her"),
    ("Chini", "our"), ("Nini", "your"), ("Norokni", "your (plural)"), ("Jotoni", "everyone's"),
]

HARD_NEGATIVE_INITIAL_WORDS = [
    "Jwngjal", "Swnamdi", "Khwlaidi", "Phwlaidi", "Naidi", "Thangdi", "Phaidi", "Swrwngdi",
    "Chaddi", "Saidi", "Swidi", "Ridi", "Tabuk", "Tini", "Khwnwi", "Bisi", "Sal",
    "Hor", "Phung", "Sanja", "Boni", "Bo", "Ang", "Chung", "Norok", "Nung",
    "Bank", "Hospital", "College", "School", "Office", "Software", "Sensor", "Signal",
    "Filter", "Register", "Social", "Media", "Internet", "Computer", "Mobile", "Message",
    "Gunda", "Police", "Doctor", "Teacher", "Engineer", "Leader", "Worker", "Order", "Rules"
]

all_sentences = []

# A. Load Gold Train
with open(GOLD_TRAIN_FILE, "r", encoding="utf-8") as f:
    curr = []
    for line in f:
        l = line.strip()
        if not l:
            if curr:
                all_sentences.append(curr)
                curr = []
        else:
            p = l.split("\t")
            if len(p) == 2:
                curr.append((p[0], p[1]))
    if curr:
        all_sentences.append(curr)

print(f"Loaded {len(all_sentences)} gold train sentences.")

def tag_span(text, label):
    toks = text.split()
    return [(toks[i], f"{'B' if i==0 else 'I'}-{label}") for i in range(len(toks))]

def O_span(*words):
    return [(w, "O") for w in words]

# B. Positive Entity Sentences (Diverse Frames) - ~1,200 sentences
for _ in range(300):
    # PER
    p = random.choice(ENTITIES["PER"])
    all_sentences.append(tag_span(p, "PER") + O_span("hwnei", "koroksa", "khamani", "kok", "sawikha", "."))
    all_sentences.append(O_span("Nini", "mung") + tag_span(p, "PER") + O_span("da", "?"))

for _ in range(400):
    # LOC (with and without clitics)
    loc = random.choice(ENTITIES["LOC"])
    all_sentences.append(tag_span(loc, "LOC") + O_span("song", "advent", "Christmas", "khlaina", "thangwi", "tongo", "."))
    all_sentences.append(O_span("Belai", "watwi", "wama", "bagwi") + tag_span(loc, "LOC") + O_span("twilom", "kha", "."))
    all_sentences.append(O_span("Bo", "song") + tag_span(loc, "LOC") + O_span("ni", "luku", "rok", "kaham", "tongo", "."))

for _ in range(300):
    # ORG (including compound distinction)
    org = random.choice(ENTITIES["ORG"])
    all_sentences.append(tag_span(org, "ORG") + O_span("Bwchwlwi", "magwnang", "bakkha", "."))
    all_sentences.append(O_span("Nini", "mokol", "kwrwi") + tag_span(org, "ORG") + O_span("channelo", "naidi", "."))
    all_sentences.append(tag_span(org, "ORG") + O_span("ni", "employee", "rok", "meeting", "khwlaikha", "."))

for _ in range(150):
    # MISC
    m = random.choice(ENTITIES["MISC"])
    all_sentences.append(tag_span(m, "MISC") + O_span("sal-o", "jotono", "hambai", "rwjakha", "."))

# C. HARD NEGATIVE SENTENCES (All 'O') - 800 sentences
# 1. Common nouns with locative / genitive clitics at sentence start and medial
for _ in range(300):
    word, _ = random.choice(COMMON_CLITIC_NOUNS)
    t = random.randint(0, 3)
    if t == 0:
        all_sentences.append(O_span(word, "thangwi", "samung", "khwlaidi", "tabuk", "."))
    elif t == 1:
        all_sentences.append(O_span("Chung", "jotono", word, "kotor", "hinui", "manikha", "."))
    elif t == 2:
        all_sentences.append(O_span("Abo", word, "kaham", "kwrwi", "wngkha", "."))
    else:
        all_sentences.append(O_span(word, "bagwi", "luku", "rok", "kok", "khorok", "."))

# 2. Sentence-initial capitalized imperatives and common words
for _ in range(300):
    w = random.choice(HARD_NEGATIVE_INITIAL_WORDS)
    all_sentences.append(O_span(w, "wngkha", "teibo", "samung", "kaham", "khwlaidi", "."))
    all_sentences.append(O_span(w, "ni", "rules", "o", "chance", "da", "mano", "tabuk", "."))

# 3. Technical and Loan words in natural Kokborok syntax
for _ in range(200):
    all_sentences.append(O_span("Jwngjal", "wngkha", "Sensor", "Signal-o", "Software", "rwmani", "Filter", "ni", "bagwise", "."))
    all_sentences.append(O_span("Coffee", "Swnamdi", ",", "Register", "khulokdi", "tei", "samung", "khwlaidi", "."))
    all_sentences.append(O_span("Bank", "ni", "challanda", "tang", "kha", "de", "tabuk", "."))
    all_sentences.append(O_span("Phaidi", "Social", "mediani", "lenglamung", "nalai", "anw", "."))

# Write output file
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for sent in all_sentences:
        for tok, tag in sent:
            f.write(f"{tok}\t{tag}\n")
        f.write("\n")

tags = [t for s in all_sentences for tok, t in s]
print(f"Generated {len(all_sentences)} sentences in {OUT_FILE.name}")
print(f"Tag distribution: {Counter(tags)}")
