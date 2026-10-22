# Kokborok Experiment 6 — ORG Data Discovery & Annotation Report

## 1. Executive Summary & Objective

The objective of **Experiment 6** is to investigate whether additional high-quality Kokborok Organization (`ORG`) examples can be discovered, verified, and curated from public resources to solve the severe ORG sparsity in the current Kokborok NER dataset (which contained only 7 gold training ORG mentions).

### Frozen Baseline Status
- **Locked Baseline Checkpoint**: [`checkpoints/kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt) (Strictly preserved, **DO NOT MODIFY**)
- **Locked Official Test Performance**:
  - Precision: **77.78%**
  - Recall: **87.50%**
  - Entity-Level F1: **82.35%**
  - Test Confusion Matrix: $\text{TP} = 14, \text{FP} = 4, \text{FN} = 2$
  - Test Token Accuracy: **99.55%**

---

## 2. Public Sources Investigated & Licenses

We systematically examined **5 primary public Kokborok corpora** spanning over **20,300 sentences**:

| Source Identifier | Repository / URL | License | Domain / Content | Total Sentences Inspected |
| :--- | :--- | :--- | :--- | :--- |
| **SpeeD-TB Kokborok** | [HF: speed-tb/kokborok](https://huggingface.co/datasets/speed-tb/kokborok) | `CC-BY-NC-SA 4.0` | Speech transcripts (education, culture, health, sports) — *Gated HF access requiring interactive agreement* | N/A (Gated) |
| **Kokborok Digitalisation** | [HF: sdmy/kokborok](https://huggingface.co/datasets/sdmy/kokborok) (`smoldoc`, `smolsent`, `gatitos`) | `MIT License` | Parallel English-Kokborok documents, sentences, and lexicon | 6,078 sentences + 3,995 phrases |
| **Wikimedia Incubator** | [Wikimedia Incubator Wp/trp](https://incubator.wikimedia.org/wiki/Wp/trp/) | `CC-BY-SA 3.0 / 4.0` | Kokborok Wikipedia test encyclopedia articles | 321 sentences (26 pages) |
| **Local Curated Datasets** | `Kokborok_datasets/*.xlsx` (6 workbooks) | Academic Research License | Parallel translation & code-mixed sentences | 4,019 sentences |
| **Kokborok QA Corpus** | [`data/raw/kokborok/qa_train.parquet`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/raw/kokborok/qa_train.parquet) | Academic Research License | Conversational and factual Question-Answering pairs | 9,886 prompt/completions |
| **TOTAL** | — | — | **Unified Public Corpus** | **20,300 sentences** |

---

## 3. Extraction, Verification & Quality Control

Every candidate was extracted and evaluated using strict linguistic criteria:
1. **Contextual Evidence**: Mentions were validated against source text semantics, translations, and syntactic roles (subject/object of organizational actions).
2. **Rejection of Non-Entities**:
   - Polysemous terms (e.g. *Amazon bolong* referring to the Amazon rainforest/river was rejected as `LOC`; *Apple buphang/thol* referring to apple trees/fruits was rejected).
   - Common nouns without proper names (e.g. *kaisa company*, *office o*, *police station*) were rejected as `O`.
   - Untranslated/pure English artifacts were filtered.
3. **Data Leakage Elimination**: Strict programmatic exclusion of any sentence overlapping with the locked test set.

### Discovery & Filtering Breakdown

| Metric | Count |
| :--- | :--- |
| **Total Sentences Inspected** | **20,300** |
| **Raw Keyword/Pattern Matches** | **3,148** |
| **Rejected Candidates (Common nouns, LOC polysemy, English noise)** | **3,080** |
| **Accepted High-Confidence ORG Instances** | **68** |
| **Distinct Unique Organizations Discovered** | **34** |
| **Candidate Dataset Size Range** | **51–100 examples** (Exact: 68) |

---

## 4. Entity & Organization Type Distribution

The 68 accepted instances provide comprehensive coverage across **7 diverse organization sectors**:

| Organization Sector | Instance Count | Percentage | Exemplar Entities Discovered |
| :--- | :--- | :--- | :--- |
| **Company / Commercial** | 26 | 38.2% | *Google, Microsoft, Apple, Amazon, Tata Technologies, Tata, SBI, ONGC, TSECL* |
| **Government / Judicial / Administrative** | 25 | 36.8% | *United Nations, Supreme Court, Indian National Congress, RBI, TBSE, Tripura Board, ISRO, NASA, Constituent Assembly, UN General Assembly, Council of Europe, health department, social welfare department* |
| **Cultural & Heritage** | 6 | 8.8% | *UNESCO (World Heritage & Intangible Cultural Heritage)* |
| **Education & Academic** | 5 | 7.4% | *Tripura University, University of Cambridge, MBB College, ITIs, Graduate Council* |
| **Media & Broadcasting** | 2 | 2.9% | *All India Radio (Akashvani), Doordarshan* |
| **Healthcare & Humanitarian** | 2 | 2.9% | *World Health Organization (WHO)* |
| **NGO / Social / Student Federations** | 2 | 2.9% | *Scheduled Castes Federation, TSF (Twipra Students' Federation)* |
| **TOTAL** | **68** | **100.0%** | **34 Distinct Organizations** |

---

## 5. Hard Cases Analysis

### 5.1 Location vs. Organization (LOC / ORG Ambiguity)
Toponymic words frequently act as locations or institutional prefixes:
- **`Tripura` as LOC**: `Tripura haste ni joto ni swlai kotor...` (*Tripura state's largest...*) $\rightarrow$ `B-LOC`
- **`Tripura` as ORG**: `Tripura University buphru sakjakha?` (*When was Tripura University established?*) $\rightarrow$ `B-ORG I-ORG`
- **`Amazon` as LOC**: `Brazil bini kotor population, kotor economy, tei Amazon bolong hai...` (*Amazon rainforest*) $\rightarrow$ Rejected from ORG.
- **`Amazon` as ORG**: `...e-commerce platform jemon Amazon, Flipkart bai...` $\rightarrow$ `B-ORG`

### 5.2 Lowercase Common Noun Organization Names
Organizations containing uncapitalized nouns that were previously misclassified or truncated:
- `health department` $\rightarrow$ `B-ORG I-ORG`
  - *Sentence*: `Bono health department tei social welfare department bai malaidi hwnwi sadi`
- `social welfare department` $\rightarrow$ `B-ORG I-ORG I-ORG`
  - *Sentence*: `Bono health department tei social welfare department bai malaidi hwnwi sadi`

### 5.3 Multi-Token Organization Examples
**25 out of 68 instances (36.8%)** are multi-token organizations:
1. `Tripura State Electricity Corporation Ltd.` (`B-ORG I-ORG I-ORG I-ORG I-ORG`)
2. `Indian National Congress` (`B-ORG I-ORG I-ORG`)
3. `Scheduled Castes Federation` (`B-ORG I-ORG I-ORG`)
4. `United Nations General Assembly` (`B-ORG I-ORG I-ORG I-ORG`)
5. `University of Cambridge` (`B-ORG I-ORG I-ORG`)
6. `World Health Organization` (`B-ORG I-ORG I-ORG`)
7. `Reserve Bank of India` (`B-ORG I-ORG I-ORG I-ORG`)
8. `State Bank of India` (`B-ORG I-ORG I-ORG I-ORG`)
9. `Tripura University` (`B-ORG I-ORG`)
10. `MBB College` (`B-ORG I-ORG`)
11. `Tripura Board` (`B-ORG I-ORG`)
12. `All India Radio` (`B-ORG I-ORG I-ORG`)

---

## 6. Data Leakage & Overlap Audit

A strict programmatic audit was conducted against all gold dataset splits:

| Gold Dataset Split | Exact Sentence Overlap Count | Status |
| :--- | :--- | :--- |
| **Gold Test (`data/gold/kokborok/test.conll`)** | **0** | **PASSED (Strict Zero Leakage Verified)** |
| **Gold Dev (`data/gold/kokborok/dev.conll`)** | **0** | **PASSED (Zero Overlap)** |
| **Gold Train (`data/gold/kokborok/train.conll`)** | **1** | Tracked |

> [!IMPORTANT]
> The locked test set was inspected solely for overlap detection. Zero sentences or entities from the test set were extracted, modified, or included in candidate sets.

---

## 7. Generated Artifacts

1. **Candidate CoNLL File**:
   [`data/processed/kokborok_org_candidates.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_org_candidates.conll)
   - Standard BIO format (`B-ORG`, `I-ORG`, `O`)
   - 68 sentences, 68 `B-ORG` tokens, 41 `I-ORG` tokens, 1,260 `O` tokens.
2. **Provenance Tracking File**:
   [`data/processed/kokborok_org_candidates_provenance.csv`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_org_candidates_provenance.csv)
   - Columns: `source`, `source_url`, `license`, `document_id`, `sentence_id`, `sentence`, `entity`, `entity_type`, `annotation_basis`, `confidence`.

---

## 8. Final Decision Gate

The curated dataset provides **68 high-confidence, verified Kokborok ORG instances** across **34 distinct organizations** in 7 key domains, expanding the available ORG training signal nearly **10-fold** (from 7 to 75 instances) with **0.00% test set overlap**.

```text
SUFFICIENT DATA — ORG augmentation experiment is justified.
```

*(Note: In accordance with strict experimental protocols, NO model training was performed, and `checkpoints/kokborok_exp4_best.pt` remains frozen and untouched).*
