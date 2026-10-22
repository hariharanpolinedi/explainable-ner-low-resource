# Comprehensive Dataset & Corpus Statistics

**Paper Section Target**: Methodology / Data & Corpus Preparation  
**Target Languages**: Konkani (`gom`, Devanagari script), Assamese (`as`, Eastern Nagari script), Sinhala (`si`, Sinhala script)  
**Constraint**: Strictly Zero Pretrained Tokenizers, Embeddings, or External Annotators  

---

## 1. Raw Corpus Consolidation & Filtering (Phase 1)

*Source: [`reports/corpus_stats.md#L9-L37`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/corpus_stats.md#L9-L37)*

| Language | Raw Harvested Sentences | Filtered Clean Sentences | Filtered Total Tokens | Unique Surface Vocab | 50,000 Sentence Floor Status | Primary Source Breakdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Konkani** | 112,235 | **51,488** | 4,950,107 | 21,176 | **MET** (>51k / 50k floor) | Wikipedia: 30,798 sents<br>News (*Bhaangarbhuin*): 20,670 sents<br>Other: 20 sents |
| **Assamese** | 119,265 | **55,002** | 4,376,034 | 25,288 | **MET** (>55k / 50k floor) | Wikipedia: 54,893 sents<br>News: 92 sents<br>Other: 17 sents |
| **Sinhala** | 113,296 | **66,954** | 5,799,421 | 30,230 | **MET** (>66k / 50k floor) | Wikipedia: 66,887 sents<br>News: 52 sents<br>Other: 15 sents |

### Filtering & Purity Rules Enforced:
- **Unicode Script Purity**: Threshold $\ge 0.90$ (at least 90% of alphabetic characters must belong to the language's native Unicode script block).
- **Length Filtering**: Dropped sentences with $< 3$ tokens or excessive character length ($> 1,000$ chars).
- **Exact Deduplication**: SHA-256 sentence hash deduplication across all sources per language.
- **Boilerplate Cleanup**: Dropped navigation templates, URL snippets, and wiki category markers.

---

## 2. Gold Benchmark Splits (Phases 3 & Post-Audit Expansion)

*Source: [`reports/phase3_split_distribution.md#L9-L49`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase3_split_distribution.md#L9-L49)*

### 2.1 Initial Seed Gold Splits (60/20/20 Ratio, 1,000 Sentences/Language)

| Language | Split | Sentences | Tokens | Total Entities | PER | LOC | ORG | MISC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `train` | 600 | 9,511 | **372** | 61 | 91 | 110 | 110 |
| | `dev` (locked) | 200 | 3,173 | **125** | 20 | 31 | 37 | 37 |
| | `test` (locked) | 200 | 3,210 | **124** | 20 | 30 | 37 | 37 |
| **Assamese** | `train` | 600 | 10,134 | **416** | 55 | 237 | 44 | 80 |
| | `dev` (locked) | 200 | 3,569 | **139** | 18 | 79 | 15 | 27 |
| | `test` (locked) | 200 | 3,349 | **138** | 19 | 79 | 14 | 26 |
| **Sinhala** | `train` | 600 | 11,626 | **393** | 113 | 116 | 83 | 81 |
| | `dev` (locked) | 200 | 4,032 | **131** | 37 | 39 | 28 | 27 |
| | `test` (locked) | 200 | 3,996 | **132** | 38 | 39 | 28 | 27 |

### 2.2 Post-Audit Expanded Gold Splits (+250 Sentences to Train Only)
*Dev and Test splits remained strictly locked and unmodified.*

| Language | Split | Sentences | Tokens | Total Entities | PER | LOC | ORG | MISC | Test Set Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | `train` (expanded) | 850 | 13,710 | **592** | 144 | 168 | 149 | 131 | N/A |
| | `dev` (locked) | 200 | 3,173 | **125** | 20 | 31 | 37 | 37 | Locked |
| | `test` (locked) | 200 | 3,210 | **124** | 20 | 30 | 37 | 37 | **LOCKED (SHA-256 Invariant)** |
| **Assamese** | `train` (expanded) | 850 | 14,649 | **679** | 102 | 395 | 67 | 115 | N/A |
| | `dev` (locked) | 200 | 3,569 | **139** | 18 | 79 | 15 | 27 | Locked |
| | `test` (locked) | 200 | 3,349 | **138** | 19 | 79 | 14 | 26 | **LOCKED (SHA-256 Invariant)** |
| **Sinhala** | `train` (expanded) | 850 | 16,541 | **641** | 178 | 224 | 119 | 120 | N/A |
| | `dev` (locked) | 200 | 4,032 | **131** | 37 | 39 | 28 | 27 | Locked |
| | `test` (locked) | 200 | 3,996 | **132** | 38 | 39 | 28 | 27 | **LOCKED (SHA-256 Invariant)** |

---

## 3. Weak Supervision Silver Volume (Phase 4)

*Source: [`reports/phase4_volume_report.md#L9-L27`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase4_volume_report.md#L9-L27)*

| Language | Target Multiplier vs Gold | Silver Sentences | Total Silver Tokens | Silver PER | Silver LOC | Silver ORG | Silver MISC | Total Silver Entities | Entity Density |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | 12x | **7,200** | 90,773 | 10 | 227 | 78 | 199 | **514** | 0.07 / sent |
| **Assamese** | 12x | **7,200** | 113,715 | 357 | 1,477 | 249 | 684 | **2,767** | 0.38 / sent |
| **Sinhala** | 12x | **7,200** | 117,119 | 183 | 220 | 227 | 444 | **1,074** | 0.15 / sent |

- **Leakage Prevention**: 100% of gold sentences (`train.conll`, `dev.conll`, `test.conll`) were indexed and excluded before generating silver data.

---

## 4. Semi-Supervised Pseudo-Labeling Volume (Phase 10)

*Source: [`reports/phase10_selftraining_curve.md#L10-L45`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase10_selftraining_curve.md#L10-L45)*

| Language | Candidate Pool Source | Pseudo Sentences Added / Round | Total Pseudo Added (2 Rounds) | Confidence Thresholds | Stopping Trigger |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Konkani** | Unlabeled Raw Pool | 2,000 | **4,000** | 0.90 (R1), 0.92 (R2) | Plateau reached (R1 accepted, R2 rolled back) |
| **Assamese** | Unlabeled Raw Pool | 2,000 | **4,000** | 0.90 (R1), 0.92 (R2) | Plateau reached (R1 & R2 maintained) |
| **Sinhala** | Unlabeled Raw Pool | 2,000 | **4,000** | 0.90 (R1), 0.92 (R2) | Plateau reached (R1 & R2 rolled back) |

---

## 5. Targeted Data Augmentation (Phase 11)

*Source: [`reports/phase11_augmentation_results.md#L10-L17`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/phase11_augmentation_results.md#L10-L17)*

- **Entity Substitution Augmentation**: Replaced named entity spans with in-class gazetteer headwords matching morphological case affixes.
- **Synonym Swap**: Thesaurus-driven non-entity token replacement.
- **Class-Imbalance Weighting**: Weighted negative log-likelihood inverse to entity frequency to balance rare entity gradients.
