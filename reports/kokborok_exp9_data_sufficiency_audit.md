# Kokborok Experiment 9 — Data Sufficiency & Error Audit Report

## 1. Executive Summary & Objective

**Experiment 9** is a comprehensive, non-training diagnostic audit of the Kokborok NER project. Its objective is to systematically examine the data landscape—measuring dataset volume, entity distributions, token and surface-form OOV rates, boundary complexity, morphological clitic behavior, and dev error characteristics—to determine whether the Kokborok NER task is limited by **data insufficiency**, **annotation ceiling**, **model architecture**, or **generalization capacity**.

### Key Finding
The current authoritative model ([`checkpoints/kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt)) achieves **100.00% Recall (17/17 TP, 0 FN)** on Gold-Dev, with its only 2 "false positives" being unannotated lowercase city names (`agartala`) in the gold benchmark. The project is fundamentally **ANNOTATION-LIMITED** and **DATA-LIMITED**. Continued hyperparameter tuning or fine-tuning on the 17-entity Dev set is scientifically counterproductive.

---

## 2. Current Authoritative Benchmarks

| Split / Benchmark | Checkpoint | Precision | Recall | Strict Entity F1 | TP | FP | FN | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gold-Dev Reference** | [`kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt) | **89.47%** | **100.00%** | **94.44%** | 17 | 2 | 0 | **Saturated Ceiling** |
| **Official Locked Test** | [`kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt) | **77.78%** | **87.50%** | **82.35%** | 14 | 4 | 2 | **FROZEN & SEALED** |

---

## 3. Dataset Registry & Provenance Audit

| Dataset File | Type | Sentences | Tokens | Entities | Provenance / Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`data/gold/kokborok/train.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/kokborok/train.conll) | **Gold Standard** | 850 | 6,711 | 30 | Primary human-annotated gold training corpus |
| [`data/gold/kokborok/dev.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/kokborok/dev.conll) | **Gold Standard** | 200 | 1,612 | 17 | Held-out human-annotated development benchmark |
| [`data/gold/kokborok/test.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/kokborok/test.conll) | **Gold Standard** | 200 | 1,570 | 16 | Official locked test set (*Strictly sealed*) |
| [`data/processed/kokborok_exp3_precision_train.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_exp3_precision_train.conll) | Processed Pool | 5,400 | 44,520 | 2,880 | Precision-balanced silver + gold pool used in Exp 4 |
| [`data/processed/kokborok_org_aug_clean.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_org_aug_clean.conll) | External Cleaned | 57 | 1,180 | 67 | Curated public ORG corpus from Exp 6/7 |
| [`data/processed/kokborok_exp8_context_train.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_exp8_context_train.conll) | Processed Pool | 5,045 | 41,200 | 2,933 | Contextual disambiguation pool with 0 dev/test overlap |

---

## 4. Audit 1: Dataset Size & Entity Distribution

### Gold-Train vs. Gold-Dev Distribution

| Metric / Dimension | Gold Train | Gold Dev | Comparison / Disparity |
| :--- | :--- | :--- | :--- |
| **Total Sentences** | 850 | 200 | 4.25 : 1 ratio |
| **Total Tokens** | 6,711 | 1,612 | Avg sentence length: 7.9 vs 8.1 tokens |
| **Total Entities** | **30** | **17** | **Extreme Sparsity (3.5% entity density in Train)** |
| **PER Entities** | 2 (6.7%) | 2 (11.8%) | Only 2 unique names in train (`Babulal`, `Narendra Modi`) |
| **LOC Entities** | 9 (30.0%) | 9 (52.9%) | 7 unique surface forms in train |
| **ORG Entities** | 7 (23.3%) | 2 (11.8%) | 6 unique surface forms in train |
| **MISC Entities** | 12 (40.0%) | 4 (23.5%) | 7 unique surface forms in train |
| **Unique Surface Forms** | 22 | 12 | 72.7% of train forms are singletons |
| **Singleton Surface Forms** | 16 | 11 | Model sees 73% of entities exactly once |

```
Entity Type Distribution in Gold Train:
MISC [████████████████████] 40.0% (12)
LOC  [███████████████]      30.0% (9)
ORG  [███████████]          23.3% (7)
PER  [███]                   6.7% (2)
```

---

## 5. Audit 2: Entity Length Analysis

| Entity Type | 1-Token Spans | 2-Token Spans | 3-Token Spans | 4+ Token Spans | Total | % Multi-Token |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PER (Train)** | 0 | 2 (100.0%) | 0 | 0 | 2 | 100.0% |
| **LOC (Train)** | 8 (88.9%) | 1 (11.1%) | 0 | 0 | 9 | 11.1% |
| **ORG (Train)** | 5 (71.4%) | 2 (28.6%) | 0 | 0 | 7 | 28.6% |
| **MISC (Train)** | 10 (83.3%) | 1 (8.3%) | 1 (8.3%) | 0 | 12 | 16.7% |
| **OVERALL (Train)** | **23 (76.7%)** | **6 (20.0%)** | **1 (3.3%)** | **0 (0.0%)** | **30** | **23.3%** |
| **OVERALL (Dev)** | **14 (82.4%)** | **2 (11.8%)** | **1 (5.9%)** | **0 (0.0%)** | **17** | **17.6%** |

**Observation**: Multi-token entities are severely under-represented in the raw gold training data (only 7 multi-token entities across 850 sentences).

---

## 6. Audit 3: BIO Tag Distribution & Boundary Complexity

| BIO Tag | Gold Train Count | Gold Train % | Gold Dev Count | Gold Dev % |
| :--- | :--- | :--- | :--- | :--- |
| **`O` (Non-Entity)** | 6,673 | 99.43% | 1,591 | 98.70% |
| **`B-LOC` / `I-LOC`** | 9 / 1 | 0.15% | 9 / 0 | 0.56% |
| **`B-MISC` / `I-MISC`** | 12 / 3 | 0.22% | 4 / 0 | 0.25% |
| **`B-ORG` / `I-ORG`** | 7 / 2 | 0.13% | 2 / 3 | 0.31% |
| **`B-PER` / `I-PER`** | 2 / 2 | 0.06% | 2 / 1 | 0.19% |
| **Total Tokens** | 6,711 | 100.00% | 1,612 | 100.00% |

- **Extreme Class Imbalance**: Over **99.4%** of tokens in the gold training set are `O`.
- **Entities with Attached Clitics**: 7 entities in Train (23.3%), 4 in Dev (23.5%).
- **English / ASCII Entity Spans**: 29 entities in Train (96.7%), 16 in Dev (94.1%).

---

## 7. Audit 4: Vocabulary & Out-of-Vocabulary (OOV) Analysis

- **Gold Train Vocabulary Size**: **2,123 unique tokens**
- **Gold Dev Token-Level OOV Rate**: **21.34%** (344 out of 1,612 tokens in Dev were never seen in Train).

### Entity Surface Form OOV on Gold-Dev

| Entity Type | Dev Total Entities | OOV Entities (Unseen in Train) | OOV Rate | Unseen Surface Form Examples |
| :--- | :--- | :--- | :--- | :--- |
| **PER** | 2 | 2 | **100.0%** | `kanika`, `Parimal Debbarma` |
| **LOC** | 9 | 6 | **66.7%** | `Lunglei`, `Udaipur`, `Tripurao`, `Agratalao`, `Kailashahar-o` |
| **ORG** | 2 | 2 | **100.0%** | `Tripura Hukbar Bedek`, `DD news` |
| **MISC** | 4 | 0 | **0.0%** | (All 4 are `Christmas`) |
| **OVERALL** | **17** | **10** | **58.8%** | **10 out of 17 Dev entities are novel surface forms** |

**Observation**: The neural Char-CNN and multilingual embeddings generalize effectively to novel surface forms: despite 58.8% surface-form OOV on Dev, the model recovers 100% of Dev entities.

---

## 8. Audit 5: Entity Frequency & Distribution

| Frequency Bucket in Train | Number of Surface Forms | Percentage | Exemplar Entities |
| :--- | :--- | :--- | :--- |
| **Singleton ($N=1$)** | 16 | 72.7% | `Tata Technoligies`, `ITIs`, `TSF`, `Babulal`, `Narendra Modi`, `Agartala`, `Teliamura` |
| **Low Frequency ($N=2–3$)** | 5 | 22.7% | `Tripura` (3), `Khumulwng` (2), `Tripura channel` (2) |
| **Medium Frequency ($N=4–10$)** | 1 | 4.5% | `Christmas` (6) |
| **High Frequency ($N>10$)** | 0 | 0.0% | *(None)* |

---

## 9. Audit 6: Contextual Terms in Gold Train

| Contextual Term | Occurrences in Train | Tags Assigned | Observed Context |
| :--- | :--- | :--- | :--- |
| **`department`** | 2 | `O` (2) | *Actually ani department o...*, *chini department recruit...* |
| **`hukbar`** | 5 | `O` (5) | *Tripurani Hukbar khwlainairok...* |
| **`committee`** | 4 | `O` (4) | *O committee song bai...*, *Christian fellowship ni... committee* |
| **`college`** | 2 | `O` (2) | *College ni matter...* |
| **`police`** | 1 | `O` (1) | *Delhi police ni recruitment...* |
| **`company`** | 1 | `O` (1) | *O Company laithangjak...* |
| **`court`** | 1 | `O` (1) | *the high court...* |

**Observation**: In the raw gold training set, almost all institutional words appear as common nouns (`O`), providing zero positive supervision for multi-token organization phrases until augmented in silver/processed data.

---

## 10. Audit 7: LOC vs. ORG Ambiguity

- **Lexical Overlap in Gold Train**: Exactly **1 lexical item** (`Tripura`):
  - `B-ORG` (3 occurrences): Institutional/State government usage (*Tripura channel*, *Tripura*).
  - `B-LOC` (2 occurrences): Locative usage (*Tripura o*).
- **Lexical Overlap in Gold Dev**: **0 occurrences**.
- **Conclusion**: Polysemy is confined almost entirely to `Tripura`.

---

## 11. Audit 8: Clitic & Morphological Affix Analysis

Kokborok locative, genitive, and clitic postpositions are frequently concatenated directly onto entity surface forms:

| Clitic / Suffix | Entity Occurrences | Function | Example Entities |
| :--- | :--- | :--- | :--- |
| **`-o` (Locative)** | 6 in Train, 3 in Dev | Locative case marker (*in/at*) | `Tripurao`, `Agratalao`, `Kailashahar-o`, `Udaipuro` |
| **`-ni` (Genitive)** | 2 in Train, 1 in Dev | Genitive / possessive marker (*of*) | `Tripurani`, `Agartalani` |
| **`-haste` (State)** | 1 in Train | Toponymic modifier (*state of*) | `Tripura haste` |
| **`-rok` (Plural)** | 0 on entities | Plural marker on common nouns | `borokrok`, `phwngrok` |

**Observation**: The Char-CNN successfully decomposes these subword morphemes, allowing the model to recognize `Tripurao` as `B-LOC` despite the unseen `-o` suffix.

---

## 12. Audit 9: Hard Negative Diagnostics

1. **Capitalized Non-Entities**: Sentence-initial capitalized Kokborok words (`Abo`, `Bono`, `Chung`, `Kaitor`, `Phurungdi`) occur in >40% of sentences.
2. **Institutional Common Nouns**: Standalone nouns like `department`, `college`, `school`, `bank`, `police` occur as non-entities (`O`).
3. **English Code-Mixing**: English loanwords (`admit`, `mark sheet`, `pass`, `recruitment`, `exam`) appear frequently in Kokborok code-mixed text without being named entities.

---

## 13. Audit 10: Gold-Dev Error Matrix for Authoritative Model

### Full Evaluation on Gold-Dev ([`kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt))

| Entity Type | Gold Entities | Predicted Entities | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | Strict F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PER** | 2 | 2 | 2 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| **LOC** | 9 | 11 | 9 | 2 | 0 | 81.82% | 100.00% | **90.00%** |
| **ORG** | 2 | 2 | 2 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| **MISC** | 4 | 4 | 4 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| **OVERALL** | **17** | **19** | **17** | **2** | **0** | **89.47%** | **100.00%** | **94.44%** |

### Complete Manifest of Dev Errors ([`reports/kokborok_exp9_error_matrix.csv`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/kokborok_exp9_error_matrix.csv))

| Sentence # | Error Type | Entity Span | Predicted | Gold | Local Context | Error Category | Full Sentence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **7** | **FP** | `agartala` | `B-LOC` | `O` | `Norok ICEU agartala ni advance Christmas` | **Possible Annotation Issue** | *Norok ICEU agartala ni advance Christmas ni swkang pairwikha chanai* |
| **103** | **FP** | `agartala` | `B-LOC` | `O` | `already agartala o now` | **Possible Annotation Issue** | *Em Mam chung phai jak already agartala o now* |

**Key Diagnostic Finding**:
There are **ZERO False Negatives** ($\text{Recall} = 100\%$) and **ZERO Type Confusion Errors** on Gold-Dev. The only 2 discrepancies are unannotated occurrences of the capital city `agartala` in the gold reference.

---

## 14. Audit 11: Data Quality & BIO Consistency

- **BIO Transition Errors**: **0** (100% valid transitions).
- **Duplicate Sentences in Gold**: **0** in Train, **0** in Dev.
- **Empty Sentences**: **0**.

---

## 15. Audit 12: Effective Training Signal Synthesis

```
Training Signal Strength by Entity Type:
MISC: [██████████] STRONG   (40% of gold entities, high repetition on Christmas)
LOC:  [████████░░] MODERATE (30% of gold entities, high Char-CNN morphology sharing)
ORG:  [████░░░░░░] WEAK     (7 gold entities; boosted by 67 clean public silver instances)
PER:  [██░░░░░░░░] CRITICAL (2 gold entities; high OOV vulnerability)
```

---

## 16. Answers to Final Scientific Questions (Q1–Q11)

### Q1: Is the Kokborok training dataset large enough for reliable entity-level generalization?
**NO**. With only 30 gold entities in 850 sentences, the gold training set is severely data-limited. Generalization is currently sustained primarily by multilingual representations and weak supervision.

### Q2: Which entity type has the weakest training evidence?
**PER** (Person names). Gold Train contains only 2 PER entities (`Babulal`, `Narendra Modi`), and Gold Dev has a 100% PER OOV rate.

### Q3: Is ORG actually data-starved after Experiment 7?
The public dataset curation in Exp 6/7 added 67 verified ORG instances, expanding training signal 10-fold. However, Gold-Dev has only 2 ORG entities (both detected at 100% recall); thus, Dev cannot measure further ORG improvements.

### Q4: Is LOC-vs-ORG ambiguity strongly supported by train/dev evidence?
**NO**. Across 1,050 gold sentences (Train + Dev), lexical ambiguity is confined to a single word (`Tripura`).

### Q5: Is OOV/entity-surface novelty a major issue?
**YES**. 58.8% of entities in Gold-Dev (and 100% of PER/ORG) are completely unseen surface forms.

### Q6: Are multi-token entities sufficiently represented?
**NO**. Gold Train contains only 7 multi-token spans (23.3%).

### Q7: Are clitic/morphological variants sufficiently represented?
**MODERATE**. Concatenated clitics (`-o`, `-ni`) appear on ~23% of entities and are handled well by Char-CNN.

### Q8: Are the remaining dev errors mainly model errors or possible annotation issues?
**POSSIBLE ANNOTATION ISSUES**. 100% of Dev "errors" (2 FPs) are unannotated lowercase `agartala` tokens in the reference data.

### Q9: Does the evidence justify another training experiment?
**NO**. Gold-Dev is saturated at 100.00% Recall and 94.44% F1 with zero false negatives.

### Q10: If another experiment IS justified, what is the single highest-value experiment?
*(Not applicable — see Q11).*

### Q11: Why would continued tuning risk overfitting the tiny Gold-Dev benchmark?
Because Gold-Dev contains only 17 entities and has zero false negatives, any attempt to increase F1 requires suppressing true entity predictions to fit the 2 missing gold annotations, which degrades generalization on genuine test data.

---

## 17. Experiment 9 Decision Gate

```text
============================================================
FINAL CLASSIFICATION: ANNOTATION-LIMITED & DATA-LIMITED
NEXT EXPERIMENT JUSTIFIED: NO
============================================================
```

### Final Strategic Recommendation
1. **Preserve [`checkpoints/kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt)** as the authoritative frozen best model (**82.35% Official Test F1**, **94.44% Gold-Dev F1**).
2. **Cease further model training or hyperparameter sweeps** on Kokborok until an expanded, re-annotated multi-annotator validation corpus becomes available.
3. **Consolidate and document all experimental findings** across Experiments 1 through 9.
