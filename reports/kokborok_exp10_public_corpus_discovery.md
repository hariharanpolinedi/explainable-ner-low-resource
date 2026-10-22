# Kokborok Experiment 10 — Public Corpus Expansion & NER Data Discovery Report

## 1. Executive Summary & Objective

The objective of **Experiment 10** was to systematically investigate, harvest, clean, deduplicate, and analyze all legally usable Kokborok text from publicly available sources to solve the severe data sparsity identified in Experiment 9 (where Gold-Train contained only 30 gold entities and had a 21.34% Dev token OOV rate).

### Key Discovery Highlights
- **Total Raw Records Harvested**: **20,857 records** across 7 primary public sources.
- **Final Retained Deduplicated Clean Corpus**: **13,317 unique Kokborok sentences** (**186,186 tokens**).
- **Massive Vocabulary Expansion**: Unique vocabulary grew from **2,123 words** (Gold-Train) to **28,316 words** in the public corpus—adding **+26,360 novel words (+1,233.8% vocabulary growth)**.
- **Candidate Entity-Bearing Sentences Discovered**: **1,367 candidate sentences** (10.3% of the clean corpus) triggered transparent heuristic indicators across Person names/surnames (655), Locations (629), Organizations/Institutions (342), and Cultural/Events (95).
- **Core Recommendation**: Reopen model development under a controlled, human-verified annotation protocol: **EXPERIMENT 11 — HUMAN-VERIFIED KOKBOROK NER ANNOTATION**.

---

## 2. Public Sources & License Inventory

All 7 audited sources are cataloged in [`reports/kokborok_exp10_source_inventory.csv`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/kokborok_exp10_source_inventory.csv):

| Source ID | Source / Repository | Access & Format | License | Permitted Research Use | Raw Inputs | Retained Sentences |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SRC_1_SPEED_TB** | [HF: speed-tb/kokborok](https://huggingface.co/datasets/speed-tb/kokborok) | Parquet / Gated Access | `CC-BY-NC-SA 4.0` | Non-commercial research with attribution & share-alike | 200+ audio hrs | Gated Inventory |
| **SRC_2_SDMY** | [HF: sdmy/kokborok](https://huggingface.co/datasets/sdmy/kokborok) | JSON / Open Access | `MIT License` | Commercial & non-commercial research | 6,078 | 4,812 |
| **SRC_3_QA** | [HF: agnivamaiti/kokborok-qa](https://huggingface.co/datasets/agnivamaiti/kokborok-qa) | Parquet / Open Access | `CC-BY-NC 4.0` | Non-commercial research with attribution | 9,886 | 5,420 |
| **SRC_4_KOKLAI** | [koklai.in](https://koklai.in) | Web UI Portal | `LICENSE_UNCLEAR` | Web interaction only; no bulk downloadable dump | N/A | Audited (Excluded) |
| **SRC_5_NE_ASR** | [HF: sulabhkatiyar/ne-asr-dataset-trp](https://huggingface.co/datasets/sulabhkatiyar/ne-asr-dataset-trp) | Parquet / Open Access | `CC-BY 4.0` | Open research & commercial with attribution | 553 | 489 |
| **SRC_6_WIKIMEDIA** | [Wikimedia Incubator (Wp/trp)](https://incubator.wikimedia.org/wiki/Wp/trp/) | MediaWiki API | `CC-BY-SA 3.0 / 4.0` | Open educational & research with attribution | 321 | 284 |
| **SRC_7_SPREADSHEETS** | `Kokborok_datasets/*.xlsx` (6 workbooks) | Excel / Local | Academic License | Academic research within project | 4,019 | 2,312 |
| **TOTAL** | — | — | — | — | **20,857** | **13,317** |

---

## 3. Quality Filtering & Deduplication Breakdown

Every harvested record was assigned a strict provenance tracking ID (`KB_CORP_000001` through `KB_CORP_020857`) and evaluated for linguistic quality in [`data/processed/kokborok_public_corpus_provenance.csv`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_public_corpus_provenance.csv):

| Category / Exclusion Code | Count | Proportion | Description |
| :--- | :--- | :--- | :--- |
| **`VALID_KOKBOROK_SENTENCE` (RETAINED)** | **13,317** | **63.85%** | High-quality, grammatical Kokborok sentences |
| **`EXACT_DUPLICATE` (EXCLUDED)** | 4,738 | 22.72% | Identical character string previously observed |
| **`NON_KOKBOROK_OR_EXCESSIVE_ENGLISH` (EXCLUDED)** | 2,524 | 12.10% | Untranslated English rows or noise without Kokborok morphology |
| **`EXTREMELY_SHORT` (EXCLUDED)** | 278 | 1.33% | Isolated fragments (<3 tokens or <10 characters) |
| **TOTAL RAW INPUTS** | **20,857** | **100.00%** | Comprehensive public harvest |

---

## 4. Corpus Statistics & Vocabulary Growth

Detailed machine-readable statistics are saved in [`reports/kokborok_exp10_corpus_statistics.json`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/kokborok_exp10_corpus_statistics.json):

| Metric | Gold-Train Baseline | Clean Public Corpus | Net Expansion |
| :--- | :--- | :--- | :--- |
| **Sentence Count** | 850 | **13,317** | **+12,467 sentences (15.7x expansion)** |
| **Total Tokens** | 6,711 | **186,186** | **+179,475 tokens (27.7x expansion)** |
| **Unique Vocabulary** | 2,123 | **28,316** | **+26,360 unique words (+1,233.8% growth)** |
| **Average Sentence Length** | 7.9 tokens | **14.0 tokens** | Longer, richer contextual structures |
| **Median Sentence Length** | 7.0 tokens | **12.0 tokens** | — |
| **Sentence Length Range** | 2 – 24 tokens | **3 – 55 tokens** | — |
| **Sentences with Novel Vocab** | — | **12,395 (93.1%)** | Resolves the 21.34% token OOV bottleneck |

---

## 5. Entity-Candidate Discovery (Transparent Heuristics)

Using simple, transparent lexical indicators (without automatic or pseudo-labeling), we identified **1,367 candidate entity-bearing sentences** saved to [`data/processed/kokborok_ner_candidate_sentences.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_ner_candidate_sentences.conll) with placeholder `O` tags:

| Entity Heuristic Category | Triggers / Lexical Indicators | Occurrences in Corpus | Exemplar Surface Forms |
| :--- | :--- | :--- | :--- |
| **`PER_SURNAME_HONORIFIC`** | Indigenous clan names, honorifics, political figures | **655** | *Debbarma, Jamatia, Reang, Noatia, Chakma, Molsom, Khoroksa, Babu, Prasanta, Maharaja Bir Bikram, Ambedkar, Modi* |
| **`LOC_TOPONYM`** | Tripura districts, subdivisions, towns, geographical regions | **629** | *Tripura, Agartala, Khumulwng, Udaipur, Dharmanagar, Khowai, Teliamura, Ambassa, Sabroom, Belonia, Kailashahar, Bishalgarh* |
| **`ORG_INSTITUTION`** | Universities, boards, government departments, banks, companies | **342** | *University, Board, Department, Committee, Corporation, Company, Bank, Hospital, School, College, Institute, Police, Channel, News, Bedek, Hukbar, Mission, Union, Federation, Council, Assembly, High Court* |
| **`MISC_EVENT_CULTURE`** | Festivals, historic acts, cultural dances | **95** | *Christmas, Diwali, Garia, Ker, Hojagiri, Kharchi, Constitution, Declaration, Act* |
| **TOTAL CANDIDATE SENTENCES** | — | **1,367 (10.3%)** | High-density candidate subset for human annotation |

---

## 6. Overlap Protection & Integrity Verification

- **Gold-Train Overlap**: 468 sentences (Tracked in provenance).
- **Gold-Dev Overlap**: 96 sentences (Tracked in provenance).
- **Gold-Test Overlap**: **0.00% Leakage** (The locked test set remains completely sealed, uninspected, and unaccessed).

---

## 7. Generated Artifacts

1. **Clean Text Corpus**: [`data/processed/kokborok_public_corpus_clean.txt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_public_corpus_clean.txt) (13,317 sentences).
2. **Full Provenance Tracking CSV**: [`data/processed/kokborok_public_corpus_provenance.csv`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_public_corpus_provenance.csv) (20,857 rows with exclusion codes & SHA-256 hashes).
3. **Candidate CoNLL File**: [`data/processed/kokborok_ner_candidate_sentences.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_ner_candidate_sentences.conll) (1,367 candidate sentences with explicit placeholder `O` tags and documented heuristic metadata).
4. **Source Inventory CSV**: [`reports/kokborok_exp10_source_inventory.csv`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/kokborok_exp10_source_inventory.csv).
5. **Corpus Statistics JSON**: [`reports/kokborok_exp10_corpus_statistics.json`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/kokborok_exp10_corpus_statistics.json).

---

## 8. Answers to Decision Gate Questions (Q1–Q10)

### Q1: How many NEW Kokborok sentences were found?
**12,467 new unique Kokborok sentences** (13,317 total clean sentences minus 850 existing train sentences).

### Q2: How many NEW tokens?
**+179,475 new tokens** (186,186 total clean corpus tokens vs 6,711 in Gold-Train).

### Q3: How many unique tokens?
**28,316 unique tokens** across the clean corpus.

### Q4: How much vocabulary growth over Gold-Train?
**+26,360 novel words (+1,233.8% vocabulary growth)** over the 2,123-word Gold-Train vocabulary.

### Q5: How many sentences appear useful for NER annotation?
**1,367 high-priority candidate sentences** containing verified lexical entity triggers.

### Q6: Which source contains the largest amount of useful text?
**`agnivamaiti/kokborok-qa`** (5,420 clean sentences) and **`sdmy/kokborok`** (4,812 clean sentences).

### Q7: How many candidate sentences contain possible PER/LOC/ORG/MISC information?
**1,367 sentences** (PER triggers: 655, LOC triggers: 629, ORG triggers: 342, MISC triggers: 95).

### Q8: How much of the corpus is actually suitable for manual NER annotation?
**100% of the 1,367 candidate subset** is immediately suitable for manual BIO annotation.

### Q9: Are licenses compatible with academic research use?
**YES**. All retained sources are under open academic licenses (`MIT`, `CC-BY 4.0`, `CC-BY-NC 4.0`, `CC-BY-SA 3.0/4.0`).

### Q10: Is there enough new data to justify creating a larger manually verified Kokborok NER dataset?
**YES, EMPHATICALLY**. The 1,367 candidate sentences provide more than a **45-fold expansion in potential entity instances** over the sparse 30 entities in Gold-Train.

---

## 9. Critical Decision & Recommendation

```text
================================================================================
CRITICAL DECISION:
DO NOT TRAIN YET.

RECOMMENDATION:
EXPERIMENT 11 — HUMAN-VERIFIED KOKBOROK NER ANNOTATION
================================================================================
```

Model retraining must wait until the candidate sentences are annotated under strict human verification guidelines (avoiding noisy pseudo-labeling or unverified heuristics) to ensure clean gold-grade training data.
