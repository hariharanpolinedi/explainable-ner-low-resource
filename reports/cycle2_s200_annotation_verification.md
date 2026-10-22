# Reliability Cycle 2 — Step 2c: Sentence #200 Annotation Verification Report

> **Task**: Guideline-referenced adjudication of canonical target Sentence #200 in the Assamese stress evaluation set (`long_multientity_eval_v2.conll`).  
> **Target Sentence**: Sentence #200 (Sentence 0 of `long_multientity_eval_v2.conll`, 42 tokens).  
> **Investigated Phenomenon**: Model prediction `[ORG: ভাৰতীয় ষ্টেট]` ("Indian State") vs Gold annotation `ভাৰতীয়/B-LOC`, `ষ্টেট/O`, `বেংক/B-ORG`.  
> **Primary Question**: Is the model prediction a genuine hallucination/error supporting Step 2b's ceiling finding, or is Sentence #200's gold label an annotation error violating project guidelines?  
> **Status**: **COMPLETE** (Adjudication resolved; gold label confirmed invalid per guidelines; candidates rescored).

---

## 1. Executive Summary & Core Determination

1. **Definitive Determination — Gold Label is an Annotation Error**:
   Sentence #200's original gold annotation for tokens 8–20 is an **unequivocal, guideline-violating annotation inconsistency**. Specifically:
   - The three real-world institutions mentioned are **Reserve Bank of India** (`ভাৰতীয় ৰিজাৰ্ভ বেংক`), **State Bank of India** (`ভাৰতীয় ষ্টেট বেংক`), and **Life Insurance Corporation of India** (`ভাৰতীয় জীৱন বিমা নিগম`).
   - The original gold annotation fragmented each into disconnected sub-spans (`B-LOC`, `O`, `B-ORG`), dropping inner constituent words (`ষ্টেট`, `ৰিজাৰ্ভ`, `জীৱন`, `বিমা`) to non-entity `O`, and misclassifying the initial nationality adjective `ভাৰতীয়` as an independent geographic location (`B-LOC`).
   - This directly violates **Section 1.1** (*"Maximal Non-Overlapping Spans: Annotate complete compound entity mentions without inner segmentation"*) and **Section 2.3 & 3** of [`annotation/guidelines_assamese.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/annotation/guidelines_assamese.md).
2. **Model Prediction Coherence**:
   When strengthened candidates (`reinforced_data`, `weight_2x`, `reinforced_weight2x`) decoded `[ORG: ভাৰতীয় ষ্টেট]`, their prediction was **strictly guideline-compliant at the token level** (`ভাৰতীয়/B-ORG`, `ষ্টেট/I-ORG`), matching the adjudicated gold labels **exactly (2/2 entity tokens correct)**. It was NOT an invalid hallucination.
3. **Nuanced Impact on Step 2b's "Ceiling" Conclusion**:
   - **Token Level**: The model achieved **2 genuine entity token hits** against the corrected gold label (compared to 0/8 on the flawed gold label).
   - **Exact Span Level**: The candidates still achieve **0 / 5 exact span hits**, because the terminal head noun `বেংক` was dropped to `O` due to the residual emission/transition barrier on length-42 sequences (emissions: $O$ $-0.04$ vs $I\text{-ORG}$ $-0.49$; $I\text{-ORG} \to O$ transition penalty overcome by $O$ advantage).
   - **Conclusion**: The model did **not** hallucinate; it successfully recognized and initiated the compound organization entity. However, the data/loss-level approach still faces an empirical ceiling in **completing full multi-token span boundaries** across long sequences without triggering runaway precision destruction.

---

## 2. Full Sentence Context & Surrounding Entities

Sentence #200 is a 42-token Assamese sentence detailing corporate headquarters located in Mumbai.

### Complete 42-Token Sequence & Annotations:

| Index | Token | Original Gold | Adjudicated Gold | Entity / Syntactic Role |
| :---: | :--- | :---: | :---: | :--- |
| **0** | `ভাৰতৰ` | `B-LOC` | `B-LOC` | India's (genitive location modifier) |
| 1 | `অসংখ্য` | `O` | `O` | numerous |
| 2 | `শিল্পগোষ্ঠী` | `O` | `O` | industrial conglomerates |
| 3 | `(` | `O` | `O` | open parenthesis |
| 4 | `লাৰচেন` | `O` | `O` | Larsen (Larsen & Toubro) |
| 5 | `অ্যাণ্ড` | `O` | `O` | & |
| 6 | `টব্ৰো` | `O` | `O` | Toubro |
| 7 | `,` | `O` | `O` | comma |
| **8** | `ভাৰতীয়` | `B-LOC` | **`B-ORG`** | Indian (Reserve Bank of India — RBI) |
| **9** | `ৰিজাৰ্ভ` | `O` | **`I-ORG`** | Reserve |
| **10** | `বেংক` | `B-ORG` | **`I-ORG`** | Bank |
| 11 | `,` | `O` | `O` | comma |
| **12** | `ভাৰতীয়` | `B-LOC` | **`B-ORG`** | Indian (State Bank of India — SBI) |
| **13** | `ষ্টেট` | `O` | **`I-ORG`** | State |
| **14** | `বেংক` | `B-ORG` | **`I-ORG`** | Bank |
| 15 | `,` | `O` | `O` | comma |
| **16** | `ভাৰতীয়` | `B-LOC` | **`B-ORG`** | Indian (Life Insurance Corporation of India — LIC) |
| **17** | `জীৱন` | `O` | **`I-ORG`** | Life |
| **18** | `বিমা` | `O` | **`I-ORG`** | Insurance |
| **19** | `নিগম` | `B-ORG` | **`I-ORG`** | Corporation |
| 20 | `,` | `O` | `O` | comma |
| 21 | `টাটা` | `O` | `O` | Tata |
| 22 | `গোষ্ঠী` | `O` | `O` | Group |
| 23 | `,` | `O` | `O` | comma |
| 24 | `গোদৰেজ` | `O` | `O` | Godrej |
| 25 | `গোষ্ঠী` | `O` | `O` | Group |
| 26 | `,` | `O` | `O` | comma |
| 27 | `ৰিলায়েন্স` | `O` | `O` | Reliance |
| 28 | `ইণ্ডাষ্ট্ৰিজ` | `O` | `O` | Industries |
| 29 | `সহ` | `O` | `O` | including |
| 30 | `)` | `O` | `O` | close parenthesis |
| 31 | `আৰু` | `O` | `O` | and |
| 32 | `পাঁচটা` | `O` | `O` | five |
| 33 | `ফৰচুন` | `O` | `O` | Fortune |
| 34 | `গ্লোবাল` | `O` | `O` | Global |
| 35 | `৫০০` | `O` | `O` | 500 |
| 36 | `কোম্পানিৰ` | `O` | `O` | companies' |
| 37 | `প্ৰধান` | `O` | `O` | head / chief |
| 38 | `কাৰ্যালয়` | `O` | `O` | office / headquarters |
| **39** | `মুম্বাইত` | `B-LOC` | `B-LOC` | in Mumbai (locative noun) |
| 40 | `অবস্থিত` | `O` | `O` | located |
| 41 | `।` | `O` | `O` | danda (period) |

---

## 3. Guideline Mapping & Real-World Fact Verification

### 3.1 Guideline Rules Cited
From [`annotation/guidelines_assamese.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/annotation/guidelines_assamese.md):
1. **Section 1.1 — Maximal Non-Overlapping Spans**:
   > *"Annotate complete compound entity mentions without inner segmentation."*
   - *Violation in original gold*: `ভাৰতীয় ষ্টেট বেংক` was broken into `ভাৰতীয়` [B-LOC], `ষ্টেট` [O], and `বেংক` [B-ORG]. Inner token `ষ্টেট` was segmented out as `O`, and the terminal noun `বেংক` was marked as an isolated entity.
2. **Section 2.3 — Organization (`ORG`)**:
   > *"Societies, Institutions, Universities, Commercial Enterprises, Political Parties: অসম/B-ORG সাহিত্য/I-ORG সভা/I-ORG, গুৱাহাটী/B-ORG বিশ্ববিদ্যালয়/I-ORG, অসম/B-ORG বিধানসভা/I-ORG..."*
   - *Direct parallel*: In `গুৱাহাটী বিশ্ববিদ্যালয়`, `গুৱাহাটী` is a geographical city, but as the initial constituent of the university name, it is tagged `B-ORG`. It is NEVER segmented as `গুৱাহাটী/B-LOC বিশ্ববিদ্যালয়/B-ORG`.
3. **Section 3 — Ambiguity Resolution Matrix**:
   > *`অসম চৰকাৰৰ উদ্যোগত` $\to$ `অসম/B-ORG চৰকাৰৰ/I-ORG` (Rationale: Government as political administration).*  
   > *`অসম সাহিত্য সভাত যোগ দিলে` $\to$ `অসম/B-ORG সাহিত্য/I-ORG সভাত/I-ORG` (Rationale: Institution governing body).*

### 3.2 Real-World Referent Verification
- **Reserve Bank of India (RBI)**: The official Assamese nomenclature used across all legal statutes and banking signage in Assam is **ভাৰতীয় ৰিজাৰ্ভ বেংক**. It is a single indivisible statutory central bank. Guideline tagging: `ভাৰতীয়/B-ORG ৰিজাৰ্ভ/I-ORG বেংক/I-ORG`.
- **State Bank of India (SBI)**: The official Assamese nomenclature on branch signage, passbooks, and legal instruments is **ভাৰতীয় ষ্টেট বেংক**. It is India's largest public sector bank. Guideline tagging: `ভাৰতীয়/B-ORG ষ্টেট/I-ORG বেংক/I-ORG`.
- **Life Insurance Corporation of India (LIC)**: The statutory title in Assamese is **ভাৰতীয় জীৱন বিমা নিগম**. It is a public sector life insurance corporation. Guideline tagging: `ভাৰতীয়/B-ORG জীৱন/I-ORG বিমা/I-ORG নিগম/I-ORG`.

### 3.3 Explicit Disambiguation Rule (Formulated per Project Precedent)
> **Disambiguation Rule (Institutional Adjectival Prefix)**:  
> When a proper nationality or location adjective (e.g., `ভাৰতীয়` [Indian], `অসম` [Assam]) forms the initial lexical constituent of an established public, statutory, educational, commercial, or banking institution, the multi-token sequence must be tagged as a **single, contiguous `ORG` mention** (`B-ORG I-ORG ... I-ORG`). It must **never** be segmented into an initial `B-LOC` followed by `O` or an isolated `B-ORG`.

---

## 4. Corpus-Wide Cross-Check Audit

To determine whether this three-way fragmentation pattern is an isolated anomaly or a systematic phenomenon across the corpus, we performed an exhaustive audit across all Assamese gold datasets.

### 4.1 Cross-Dataset Comparison:

| Dataset | Sentences | Total `ভাৰতীয়` Tokens | `ভাৰতীয়` as `B-LOC` | `ভাৰতীয়` as `B-ORG` | `ভাৰতীয়` as `O` | Fragmented `B-LOC ... B-ORG` Institutional Spans |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train.conll`** (Original) | 850 | 92 | **0 (0.0%)** | 13 (14.1%) | 79 (85.9%) | **0** |
| **`train_cycle2_corrected.conll`** | 850 | 92 | **0 (0.0%)** | **53 (57.6%)** | 39 (42.4%) | **0** |
| **`dev.conll`** | 200 | 18 | **0 (0.0%)** | 8 (44.4%) | 10 (55.6%) | **0** |
| **`test.conll`** | 200 | 22 | **0 (0.0%)** | 3 (13.6%) | 19 (86.4%) | **0** |
| **`supplementary_long_multientity.conll`** | 35 | 7 | **0 (0.0%)** | 0 (0.0%) | 7 (100.0%) | **0** |
| **`long_multientity_eval_v2.conll`** | 35 | 6 | **6 (100.0%)** | 0 (0.0%) | 0 (0.0%) | **5 (4 institutional, 1 syntactic)** |

### 4.2 Critical Audit Finding:
1. **Zero Precedent in Standard Gold Sets**:
   In `train.conll`, `dev.conll`, `test.conll`, and `supplementary_long_multientity.conll` combined (1,285 sentences, 139 occurrences of `ভাৰতীয়`), `ভাৰতীয়` is **NEVER tagged as `B-LOC`** (0 / 139).
   - In `train` and `dev`, institutional compounds such as `ভাৰতীয় জাতীয় কংগ্ৰেছ` (Indian National Congress) and `ভাৰতীয় জনতা পাৰ্টি` (BJP) are consistently tagged as contiguous `ORG` spans: `ভাৰতীয়/B-ORG জাতীয়/I-ORG কংগ্ৰেছ/I-ORG`.
   - There is **not a single instance** of `B-LOC O B-ORG` representing an institution in `train`, `dev`, `test`, or `supplementary_long_multientity`.
2. **Localized Benchmark Asymmetry in `long_multientity_eval_v2.conll`**:
   The fragmented pattern appears exclusively within `long_multientity_eval_v2.conll`:
   - Sentence 0 (#200): `ভাৰতীয়/B-LOC ৰিজাৰ্ভ/O বেংক/B-ORG`
   - Sentence 0 (#200): `ভাৰতীয়/B-LOC ষ্টেট/O বেংক/B-ORG`
   - Sentence 0 (#200): `ভাৰতীয়/B-LOC জীৱন/O বিমা/O নিগম/B-ORG`
   - Sentence 14: `অসম/B-LOC বিধান/O পৰিষদ/B-ORG` (Assam Legislative Council, directly violating Guidelines 2.3 `অসম/B-ORG বিধানসভা/I-ORG`).
   - Sentence 2: `মুম্বাইত/B-LOC অৱস্থিত/O ইণ্ডিয়ান/B-ORG...` (genuine syntactic boundary: "located in Mumbai, IIT...").
3. **Conclusion on Scope**:
   This is **not** a widespread corpus-level pattern requiring an expansion of Step 1's training audit. Rather, it is a **localized annotation defect in `long_multientity_eval_v2.conll`**, where a single annotator pass mislabeled compound organizational prefixes as standalone locations.

---

## 5. Re-Scoring Results for Step 2b Candidates

We re-scored all Step 2b candidates on Sentence #200 using GPU (`cuda`), evaluating exact span hits and token-level entity hits against both the original flawed gold annotation and the adjudicated gold annotation.

### Re-Scoring Summary Table:

| Candidate Checkpoint | Decoded Span on Sentence #200 | Original Gold Span Hits (out of 8) | Adjudicated Gold Span Hits (out of 5) | Original Entity Token Hits (out of 8) | Adjudicated Entity Token Hits (out of 12) | Mean $O$-Bias Gap | Token 0 Gap |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | `None (all O)` | 0 / 8 | 0 / 5 | 0 / 8 | 0 / 12 | +9.43 | +10.63 |
| **`step2_combined`** | `None (all O)` | 0 / 8 | 0 / 5 | 0 / 8 | 0 / 12 | +0.85 | +0.75 |
| **`weight_2x`** | `[ORG: ভাৰতীয় ষ্টেট]` | 0 / 8 | 0 / 5 | 0 / 8 | **2 / 12** | +2.05 | +2.74 |
| **`weight_4x`** | `None (all O)` | 0 / 8 | 0 / 5 | 0 / 8 | 0 / 12 | +2.67 | +3.38 |
| **`weight_8x`** | `None (all O)` | 0 / 8 | 0 / 5 | 0 / 8 | 0 / 12 | +3.10 | +3.83 |
| **`reinforced_data`** | `[ORG: ভাৰতীয় ষ্টেট]` | 0 / 8 | 0 / 5 | 0 / 8 | **2 / 12** | **+0.60** | **+0.61** |
| **`reinforced_weight2x`** | `[ORG: ভাৰতীয় ষ্টেট]` | 0 / 8 | 0 / 5 | 0 / 8 | **2 / 12** | +2.00 | +2.71 |

### Token-Level Decoded Logit Dynamics on Tokens 12–14 (`reinforced_data`):
- **Token 12 (`ভাৰতীয়`)**:
  - $B\text{-ORG}$ emission: **$+0.27$** vs $O$ emission: **$-0.22$** ($O$-bias gap flipped negative: $-0.49$).
  - Decoded tag: **`B-ORG`** (Matches adjudicated gold `B-ORG` **HIT**).
- **Token 13 (`ষ্টেট`)**:
  - $I\text{-ORG}$ emission: **$-0.06$** vs $O$ emission: **$-0.25$** ($O$-bias gap flipped negative: $-0.18$).
  - Decoded tag: **`I-ORG`** (Matches adjudicated gold `I-ORG` **HIT**).
- **Token 14 (`বেংক`)**:
  - $O$ emission: **$-0.04$** vs $I\text{-ORG}$ emission: **$-0.49$** (Residual positive $O$ gap: $+0.45$).
  - Decoded tag: **`O`** (Missed; CRF preferred returning to background $O$).

---

## 6. Scientific Verdict & Impact on Step 2b / Step 3 Conclusions

1. **Was the Model's Prediction an Error?**
   **No.** The prediction of `[ORG: ভাৰতীয় ষ্টেট]` was a **guideline-accurate recognition of the entity type and the initial entity boundaries**. The original evaluation metric falsely penalized the model as having 0 hits because the benchmark itself had mislabeled the token as `B-LOC` and dropped the middle word to `O`.
2. **Does this Invalidate Step 2b's "Ceiling" Conclusion?**
   **No, but it significantly refines it.**
   - Step 2b correctly identified that data and loss weighting alone cannot achieve **complete multi-token span recovery** across long sequences: the model predicts `[ORG: ভাৰতীয় ষ্টেট]` (2 tokens) instead of `[ORG: ভাৰতীয় ষ্টেট বেংক]` (3 tokens), achieving 0 exact span hits.
   - Pushing the weights further ($4\times, 8\times$) causes early-stopping to pull back to prevent runaway false-positive hallucination, while letting training run (Epoch 5 at $2\times$) destroys dev precision (51.02%).
   - Therefore, the empirical ceiling is **not** an inability of the model to recognize Assamese entities; rather, it is a **boundary-completion barrier** imposed by sequence length and the CRF transition matrix on the trailing tokens of long multi-entity sentences.
3. **Verdict for Step 3**:
   Step 3 should document that:
   - Assamese linear emission bias was reduced by over $93\%$ through label correction and class weighting.
   - The model successfully recovered genuine entity tokens on Sentence #200 (`ভাৰতীয়/B-ORG ষ্টেট/I-ORG`).
   - Complete span boundary resolution on long stress sentences requires architectural or structured inference support beyond simple loss-weight scaling.

---

## 7. Dataset Handling & Cryptographic Integrity

To preserve strict non-circularity and integrity-lock discipline:
1. **Baseline Preserved**: `data/gold/assamese/long_multientity_eval_v2.conll` was **NOT modified in place**. Its original SHA-256 hash remains:
   `bce6d69071a26e3180eae58f54e9e22cf66a9bd6389ca5aa51f26fbb5106b17c` (verified matching [`reports/long_multientity_eval_v2_lockfile.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/long_multientity_eval_v2_lockfile.md)).
2. **Versioned Adjudicated File Created**: An explicit, versioned benchmark file was generated:
   [`data/gold/assamese/long_multientity_eval_v2_s200_adjudicated.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/assamese/long_multientity_eval_v2_s200_adjudicated.conll)  
   **SHA-256 Checksum**: `e51942e8ba1cda0d3b86af11e6f2fb0bf64bbadfe63d3f677422179e72414d04`
3. **Log Entry Appended**: An adjudicated log entry with complete guideline justification was added as Section 5 of [`reports/cycle2_assamese_label_correction_log.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/cycle2_assamese_label_correction_log.md).
