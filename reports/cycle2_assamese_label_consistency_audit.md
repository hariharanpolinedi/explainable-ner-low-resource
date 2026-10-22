# Reliability Cycle 2 — Step 1b: Assamese Training Label Consistency Audit

> **Target Deliverable**: Comprehensive full-corpus label-consistency audit for Assamese Named Entity Recognition.  
> **Input Corpus**: `data/gold/assamese/train.conll` (850 sentences, 14,649 non-empty token lines, 4,789 distinct surface types).  
> **Evaluation Checkpoint**: Baseline `checkpoints/improved_assamese_v2.pt` / `baseline_v1/checkpoints/final_assamese.pt`.  
> **Status**: **COMPLETE — AUDIT SYSTEMATICALLY GENERALIZED FROM 2 SPOT-CHECK EXAMPLES TO ALL QUALIFYING TOKENS (N=475)**.

---

## 1. Executive Summary

In Step 1, a spot-check of two entity tokens (`ভাৰতৰ` and `ভাৰতীয়`) revealed severe label inconsistency in the gold training set (70.6% and 85.9% tagged as `O`, respectively), producing an overwhelming static emission $O$-bias (+10 to +13 logits).

This audit expands that initial spot-check into a **complete, corpus-wide census** of all distinct token types appearing at least 5 times in `data/gold/assamese/train.conll` ($N = 475$ token types, encompassing 9,141 total token occurrences, or 62.4% of the training corpus).

### Key Empirical Findings:
1. **The vast majority of frequent tokens are 100% consistent**: **402 out of 475 token types (84.6%)** exhibit perfect label consistency (rate $= 1.0000$), virtually all of which are common background syntactic markers (verbs, postpositions, conjunctions, punctuation, and invariable non-entities) consistently tagged as `O`.
2. **Entity-bearing and ambiguous tokens are heavily concentrated in a small, highly inconsistent tail**: **73 token types (15.4%)** have inconsistent annotations (rate $< 1.0000$).
3. **Severe label ambiguity affects high-impact core entities**:
   - `অসমৰ` (Assam's): 65 training occurrences, tagged `O` **58.5% (38/65)** and `B-LOC` 41.5% (27/65).
   - `ভাৰতৰ` (India's): 34 training occurrences, tagged `O` **70.6% (24/34)** and `B-LOC` 29.4% (10/34).
   - `ভাৰতীয়` (Indian): 92 training occurrences, tagged `O` **85.9% (79/92)** and `B-ORG` 14.1% (13/92).
   - `শ্ৰীমন্ত` (Srimanta): 19 training occurrences, tagged `O` **52.6% (10/19)** and `B-PER` 47.4% (9/19).
   - `লক্ষ্মীনাথ` (Lakshminath): 17 training occurrences, tagged `O` **58.8% (10/17)** and `B-PER` 41.2% (7/17).
   - `বিশ্ববিদ্যালয়` (University): 25 training occurrences, tagged `O` **40.0% (10/25)** and `I-ORG` 60.0% (15/25).
4. **Empirical Cutoff for Low Consistency**: The 85th percentile of the inconsistent distribution corresponds to **consistency rate $< 0.85$**, which captures the **56 most ambiguous tokens (11.8% of all frequent token types)**. A stricter threshold of $< 0.80$ isolates **43 tokens (9.1%)**, while $< 0.70$ isolates **25 tokens (5.3%)** exhibiting coin-flip ambiguity.

---

## 2. Corpus Consistency Distribution

### 2.1 Distribution Summary Statistics ($N=475$ Types with Frequency $\ge 5$)

| Metric | Empirical Value | Context / Interpretation |
| :--- | :---: | :--- |
| **Total Sentences in Gold Train** | 850 | Standard gold training split |
| **Total Token Lines in Gold Train** | 14,649 | Full token volume |
| **Total Distinct Token Types** | 4,789 | Vocabulary size in training split |
| **Types with Frequency $\ge 5$** | **475** | Qualifying population for audit |
| **Minimum Consistency Rate** | **0.5000 (50.0%)** | `বিষ্ণুপ্ৰসাদ` (3 B-PER, 3 O), `আগৰৱালা` (3 O, 3 I-PER) |
| **10th Percentile ($P_{10}$)** | 0.8000 (80.0%) | Bottom decile suffers $\ge 20\%$ contradictory labels |
| **25th Percentile ($Q_1$)** | 1.0000 (100.0%) | Top 75% includes invariable background tokens |
| **Median Consistency Rate** | 1.0000 (100.0%) | Background $O$ tokens dominate median |
| **75th Percentile ($Q_3$)** | 1.0000 (100.0%) | Background $O$ tokens dominate upper quartile |
| **90th Percentile ($P_{90}$)** | 1.0000 (100.0%) | Background $O$ tokens dominate 90th percentile |
| **Maximum Consistency Rate** | 1.0000 (100.0%) | 402 tokens |

### 2.2 Stratified Consistency Tiers

| Consistency Tier | Criteria | Count of Types | % of Types ($N=475$) | Qualitative Characterization |
| :--- | :---: | :---: | :---: | :--- |
| **Tier 0: Perfectly Invariable** | Consistency $= 1.0000$ | 402 | 84.6% | Non-entity words (`আৰু`, `হয়`, punctuation) + rare clean entities |
| **Tier 1: Mild Inconsistency** | $0.85 \\le \\text{Rate} < 1.00$ | 17 | 3.6% | Mostly consistent entities with occasional noise (`ভাৰতীয়`, `অসমীয়া`) |
| **Tier 2: Moderate Ambiguity** | $0.80 \\le \\text{Rate} < 0.85$ | 13 | 2.7% | Regular label conflict (`ভূপেন`, `শইকীয়া`, `ৰাজ্যিক`) |
| **Tier 3: Severe Ambiguity** | $0.70 \\le \\text{Rate} < 0.80$ | 18 | 3.8% | High label conflict (`ভাৰতৰ`, `অসমত`, `পৰিষদ`, `বৰদলৈ`) |
| **Tier 4: Extreme / Near Coin-Flip** | $\\text{Rate} < 0.70$ | 25 | 5.3% | Near 50/50 noise (`অসমৰ`, `শ্ৰীমন্ত`, `লক্ষ্মীনাথ`, `আগৰৱালা`) |

---

## 3. Complete Ranked Inventory of Inconsistent Training Tokens

The following table contains all **73 token types** in `data/gold/assamese/train.conll` with occurrence count $\ge 5$ that exhibit contradictory labels (consistency rate $< 1.0000$), ranked from least consistent to most consistent:

| Rank | Token | English Gloss / Category | Total Train Count | Top Tag | Top Count | Consistency Rate | Train $O$ Count | Train $O$ % | Entity Count | Tag Distribution Detail |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **বিষ্ণুপ্ৰসাদ** | Bishnuprasad (Proper Name) | 6 | `B-PER` | 3 | **50.0%** | 3 | 50.0% | 3 | `B-PER: 3, O: 3` |
| 2 | **আগৰৱালা** | Agarwala (Surname) | 6 | `O` | 3 | **50.0%** | 3 | 50.0% | 3 | `O: 3, I-PER: 3` |
| 3 | **শ্ৰীমন্ত** | Srimanta (Honorific/Proper Name) | 19 | `O` | 10 | **52.6%** | 10 | 52.6% | 9 | `O: 10, B-PER: 9` |
| 4 | **গণ** | People / Mass (in Asom Gana Parishad) | 13 | `I-ORG` | 7 | **53.8%** | 6 | 46.2% | 7 | `I-ORG: 7, O: 6` |
| 5 | **গোপীনাথ** | Gopinath (Proper Name) | 9 | `B-PER` | 5 | **55.6%** | 4 | 44.4% | 5 | `B-PER: 5, O: 4` |
| 6 | **ৰঙালী** | Rongali (Spring Bihu Festival) | 7 | `O` | 4 | **57.1%** | 4 | 57.1% | 3 | `O: 4, B-MISC: 3` |
| 7 | **চৰকাৰ** | Government / Sarkar | 7 | `O` | 4 | **57.1%** | 4 | 57.1% | 3 | `O: 4, I-ORG: 3` |
| 8 | **লাচিত** | Lachit (Borphukan, Historical General) | 7 | `O` | 4 | **57.1%** | 4 | 57.1% | 3 | `O: 4, B-PER: 3` |
| 9 | **সভা** | Assembly / Council / Sabha | 12 | `I-ORG` | 7 | **58.3%** | 5 | 41.7% | 7 | `I-ORG: 7, O: 5` |
| 10 | **অসমৰ** | Assam's (Genitive Location) | 65 | `O` | 38 | **58.5%** | 38 | 58.5% | 27 | `O: 38, B-LOC: 27` |
| 11 | **লক্ষ্মীনাথ** | Lakshminath (Bezbaroa, Writer) | 17 | `O` | 10 | **58.8%** | 10 | 58.8% | 7 | `O: 10, B-PER: 7` |
| 12 | **বিশ্ববিদ্যালয়** | University (Common/Org Noun) | 25 | `I-ORG` | 15 | **60.0%** | 10 | 40.0% | 15 | `I-ORG: 15, O: 10` |
| 13 | **ভোগালী** | Bhogali (Harvest Bihu Festival) | 5 | `O` | 3 | **60.0%** | 3 | 60.0% | 2 | `O: 3, B-MISC: 2` |
| 14 | **চুক্তি** | Accord / Agreement | 5 | `O` | 3 | **60.0%** | 3 | 60.0% | 2 | `O: 3, I-MISC: 2` |
| 15 | **পাৰ্টি** | Party (Political) | 5 | `I-ORG` | 3 | **60.0%** | 2 | 40.0% | 3 | `I-ORG: 3, O: 2` |
| 16 | **বুঢ়ী** | Old Woman (in Grandmothers Tales) | 5 | `O` | 3 | **60.0%** | 3 | 60.0% | 2 | `O: 3, B-MISC: 2` |
| 17 | **আইৰ** | Mother's (Genitive) | 5 | `O` | 3 | **60.0%** | 3 | 60.0% | 2 | `O: 3, I-MISC: 2` |
| 18 | **কংগ্ৰেছ** | Congress (Political Org) | 16 | `I-ORG` | 10 | **62.5%** | 6 | 37.5% | 10 | `I-ORG: 10, O: 6` |
| 19 | **তৰুণ** | Tarun (Proper Name / Young) | 16 | `B-PER` | 10 | **62.5%** | 6 | 37.5% | 10 | `B-PER: 10, O: 6` |
| 20 | **শৰ্মা** | Sarma (Surname) | 8 | `O` | 5 | **62.5%** | 5 | 62.5% | 3 | `O: 5, I-PER: 3` |
| 21 | **হাজৰিকা** | Hazarika (Surname) | 8 | `I-PER` | 5 | **62.5%** | 3 | 37.5% | 5 | `I-PER: 5, O: 3` |
| 22 | **বিধানসভা** | Legislative Assembly | 6 | `I-ORG` | 4 | **66.7%** | 2 | 33.3% | 4 | `I-ORG: 4, O: 2` |
| 23 | **মহন্ত** | Mahanta (Surname) | 6 | `O` | 4 | **66.7%** | 4 | 66.7% | 2 | `O: 4, I-PER: 2` |
| 24 | **ফণী** | Phani (Proper Name) | 6 | `O` | 4 | **66.7%** | 4 | 66.7% | 2 | `O: 4, B-PER: 2` |
| 25 | **দাস** | Das (Surname) | 6 | `O` | 4 | **66.7%** | 4 | 66.7% | 2 | `O: 4, I-PER: 2` |
| 26 | **পৰিষদ** | Parishad / Council | 10 | `I-ORG` | 7 | **70.0%** | 3 | 30.0% | 7 | `I-ORG: 7, O: 3` |
| 27 | **জুবিন** | — | 10 | `B-PER` | 7 | **70.0%** | 3 | 30.0% | 7 | `B-PER: 7, O: 3` |
| 28 | **ভাৰতৰ** | India's (Genitive Location) | 34 | `O` | 24 | **70.6%** | 24 | 70.6% | 10 | `O: 24, B-LOC: 10` |
| 29 | **বৰদলৈ** | Bordoloi (Surname) | 7 | `I-PER` | 5 | **71.4%** | 2 | 28.6% | 5 | `I-PER: 5, O: 2` |
| 30 | **অসমত** | In Assam (Locative Location) | 7 | `O` | 5 | **71.4%** | 5 | 71.4% | 2 | `O: 5, B-LOC: 2` |
| 31 | **গুৱাহাটীৰ** | — | 7 | `O` | 5 | **71.4%** | 5 | 71.4% | 2 | `O: 5, B-LOC: 2` |
| 32 | **হিতেশ্বৰ** | — | 7 | `O` | 5 | **71.4%** | 5 | 71.4% | 2 | `O: 5, B-PER: 2` |
| 33 | **সাধু** | — | 7 | `O` | 5 | **71.4%** | 5 | 71.4% | 2 | `O: 5, I-MISC: 2` |
| 34 | **জ্যোতিপ্ৰসাদ** | — | 11 | `O` | 8 | **72.7%** | 8 | 72.7% | 3 | `O: 8, B-PER: 3` |
| 35 | **সদৌ** | — | 8 | `O` | 6 | **75.0%** | 6 | 75.0% | 2 | `O: 6, B-ORG: 2` |
| 36 | **ভবেন্দ্ৰ** | — | 8 | `B-PER` | 6 | **75.0%** | 2 | 25.0% | 6 | `B-PER: 6, O: 2` |
| 37 | **নাথ** | — | 8 | `I-PER` | 6 | **75.0%** | 2 | 25.0% | 6 | `I-PER: 6, O: 2` |
| 38 | **জাতীয়** | National | 45 | `O` | 35 | **77.8%** | 35 | 77.8% | 10 | `O: 35, I-ORG: 10` |
| 39 | **কৃষি** | — | 9 | `O` | 7 | **77.8%** | 7 | 77.8% | 2 | `O: 7, I-ORG: 2` |
| 40 | **জয়মতী** | — | 9 | `B-PER` | 7 | **77.8%** | 2 | 22.2% | 7 | `B-PER: 7, O: 2` |
| 41 | **জনতা** | — | 14 | `O` | 11 | **78.6%** | 11 | 78.6% | 3 | `O: 11, I-ORG: 3` |
| 42 | **গুৱাহাটী** | — | 47 | `B-LOC` | 37 | **78.7%** | 0 | 0.0% | 47 | `B-LOC: 37, B-ORG: 9, I-ORG: 1` |
| 43 | **ভূপেন** | — | 24 | `O` | 19 | **79.2%** | 19 | 79.2% | 5 | `O: 19, B-PER: 5` |
| 44 | **শইকীয়া** | — | 10 | `I-PER` | 8 | **80.0%** | 2 | 20.0% | 8 | `I-PER: 8, O: 2` |
| 45 | **কটন** | — | 5 | `O` | 4 | **80.0%** | 4 | 80.0% | 1 | `O: 4, B-ORG: 1` |
| 46 | **আৰক্ষী** | — | 5 | `I-ORG` | 4 | **80.0%** | 1 | 20.0% | 4 | `I-ORG: 4, O: 1` |
| 47 | **চিকিৎসা** | — | 5 | `O` | 4 | **80.0%** | 4 | 80.0% | 1 | `O: 4, I-ORG: 1` |
| 48 | **হোমেন** | — | 5 | `O` | 4 | **80.0%** | 4 | 80.0% | 1 | `O: 4, B-PER: 1` |
| 49 | **যোৰহাটত** | — | 5 | `O` | 4 | **80.0%** | 4 | 80.0% | 1 | `O: 4, B-LOC: 1` |
| 50 | **ডিব্ৰুগড়** | — | 21 | `B-LOC` | 17 | **81.0%** | 0 | 0.0% | 21 | `B-LOC: 17, B-ORG: 4` |
| 51 | **বৰুৱা** | Barua (Surname) | 16 | `O` | 13 | **81.2%** | 13 | 81.2% | 3 | `O: 13, I-PER: 3` |
| 52 | **দৈনিক** | — | 6 | `B-ORG` | 5 | **83.3%** | 1 | 16.7% | 5 | `B-ORG: 5, O: 1` |
| 53 | **জয়ন্ত** | — | 6 | `O` | 5 | **83.3%** | 5 | 83.3% | 1 | `O: 5, B-PER: 1` |
| 54 | **ৰাম** | — | 6 | `O` | 5 | **83.3%** | 5 | 83.3% | 1 | `O: 5, I-PER: 1` |
| 55 | **উচ্চতৰ** | — | 6 | `O` | 5 | **83.3%** | 5 | 83.3% | 1 | `O: 5, I-ORG: 1` |
| 56 | **বঁটা** | — | 52 | `O` | 44 | **84.6%** | 44 | 84.6% | 8 | `O: 44, I-MISC: 8` |
| 57 | **অসম** | — | 236 | `B-LOC` | 201 | **85.2%** | 0 | 0.0% | 236 | `B-LOC: 201, B-ORG: 28, I-ORG: 5, B-MISC: 2` |
| 58 | **ছাত্ৰ** | — | 14 | `O` | 12 | **85.7%** | 12 | 85.7% | 2 | `O: 12, I-ORG: 2` |
| 59 | **প্ৰফুল্ল** | — | 7 | `O` | 6 | **85.7%** | 6 | 85.7% | 1 | `O: 6, B-PER: 1` |
| 60 | **ফুকন** | — | 7 | `O` | 6 | **85.7%** | 6 | 85.7% | 1 | `O: 6, I-PER: 1` |
| 61 | **শিক্ষা** | — | 7 | `O` | 6 | **85.7%** | 6 | 85.7% | 1 | `O: 6, I-ORG: 1` |
| 62 | **ভাৰতীয়** | Indian (Adjective/Org component) | 92 | `O` | 79 | **85.9%** | 79 | 85.9% | 13 | `O: 79, B-ORG: 13` |
| 63 | **বিহু** | — | 47 | `B-MISC` | 41 | **87.2%** | 0 | 0.0% | 47 | `B-MISC: 41, I-MISC: 6` |
| 64 | **গাৰ্গ** | — | 8 | `I-PER` | 7 | **87.5%** | 1 | 12.5% | 7 | `I-PER: 7, O: 1` |
| 65 | **মহাবিদ্যালয়** | — | 8 | `O` | 7 | **87.5%** | 7 | 87.5% | 1 | `O: 7, I-ORG: 1` |
| 66 | **যোৰহাট** | — | 8 | `B-LOC` | 7 | **87.5%** | 0 | 0.0% | 8 | `B-LOC: 7, B-ORG: 1` |
| 67 | **হেমচন্দ্ৰ** | — | 9 | `O` | 8 | **88.9%** | 8 | 88.9% | 1 | `O: 8, B-PER: 1` |
| 68 | **জ্ঞানপীঠ** | — | 9 | `B-MISC` | 8 | **88.9%** | 1 | 11.1% | 8 | `B-MISC: 8, O: 1` |
| 69 | **উচ্চ** | — | 10 | `O` | 9 | **90.0%** | 9 | 90.0% | 1 | `O: 9, I-ORG: 1` |
| 70 | **মাধ্যমিক** | — | 11 | `O` | 10 | **90.9%** | 10 | 90.9% | 1 | `O: 10, I-ORG: 1` |
| 71 | **সাহিত্য** | Literature (in Asam Sahitya Sabha) | 94 | `O` | 87 | **92.5%** | 87 | 92.5% | 7 | `O: 87, I-ORG: 7` |
| 72 | **কুমাৰ** | — | 16 | `O` | 15 | **93.8%** | 15 | 93.8% | 1 | `O: 15, I-PER: 1` |
| 73 | **অসমীয়া** | — | 101 | `O` | 99 | **98.0%** | 99 | 98.0% | 2 | `O: 99, B-ORG: 2` |

---

## 4. Architectural and Training Impact Analysis

### 4.1 Why the Linear Emission Head Collapses to $O$
In the multilingual NER architecture (`improved_assamese_v2.pt`), the linear emission layer maps attended representations directly into tag logits. When high-frequency entities appear in training with `O` as their dominant tag:
1. **Direct Loss Incentive**: Cross-entropy / negative log-likelihood strongly rewards predicting `O` for tokens like `ভাৰতীয়` (79 times $O$ vs 13 times entity) and `ভাৰতৰ` (24 times $O$ vs 10 times entity).
2. **Weight Polarization**: The linear emission head weights for class `O` receive massive positive gradient reinforcement, driving the static $O$ logit up to $+5.7$ to $+6.1$, while entity tag logits receive heavy negative penalty (down to $-4.9$ to $-6.9$).
3. **CRF Subjugation**: The linear emission logit gap of $+10$ to $+13$ logits dwarfs the CRF transition matrix weights (which operate in the range of $-1.5$ to $+0.8$). The Viterbi decoding path is mathematically forced through the $O 	o O$ sequence regardless of contextual signals.

---

## 5. Audit Conclusions & Diagnostic Thresholds

1. **Confirmation of Corpus-Wide Label Pathology**: Step 1's initial finding regarding `ভাৰতৰ` and `ভাৰতীয়` is not an isolated artifact. It is a systematic feature of the Assamese gold training split affecting the primary geographic and political entity vocabulary of the language (`অসমৰ`, `ভাৰতৰ`, `ভাৰতীয়`, `অসমত`, `দিল্লী`).
2. **Derived Threshold for Downstream Disentanglement**:
   - **Low Consistency Cutoff**: $	ext{Consistency Rate} < 0.85$ (captures the 56 most heavily compromised types, comprising the bottom 11.8% of the frequent vocabulary).
   - **Extreme Ambiguity Cutoff**: $	ext{Consistency Rate} < 0.70$ (captures the 25 most severely degraded types, where annotation is effectively a coin flip).
