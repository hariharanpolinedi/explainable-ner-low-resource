# Kokborok Data Sufficiency & Pipeline Adaptation Decision Report

**Phase**: 13 — Kokborok Training & Standalone Evaluation  
**Auditor**: Senior ML/NLP Research Engineer  
**Status**: COMPLETE (Authoritative Empirical Finding)  
**Date**: 2026-09-25  

---

## 1. Executive Summary: Evidence-Driven Decision on Data Scaling

Before committing to the final Kokborok training run, a rigorous data-sufficiency audit was performed following the methodology established in Phase 3's learning-curve study for Konkani, Assamese, and Sinhala.

### Core Question
Does Kokborok's data situation warrant the identical self-training/augmentation pipeline (large silver weak-supervision corpus, pseudo-labeling) used for the original three languages, or does Kokborok's empirical data situation require an adapted, scaled strategy?

### Authoritative Finding & Decision
**Self-training / weak-supervision expansion on the unannotated Kokborok corpus is STRICTLY REJECTED; training must proceed strictly on the curated Gold-Train split under the protected modular Option (c) architecture.**

---

## 2. Comparative Data Volume & Entity Density Audit

| Metric | Original 3 Languages (Konkani / Assamese / Sinhala) | Kokborok (`trp`) | Comparison Factor |
| :--- | :---: | :---: | :---: |
| **Raw Unannotated Corpus Size** | Tens of thousands of sentences (News / Wikipedia) | **4,020 sentences** ($3,988$ deduplicated) | ~10x–20x smaller raw corpus |
| **Gold Split Sentences (Train / Dev / Test)** | 850 / 200 / 200 | **850 / 200 / 200** | Identical sentence split size |
| **Gold Train Tokens** | 13,710 (Kon) / 14,649 (Asm) / 16,541 (Sin) | **6,711 tokens** | ~2x–2.5x shorter sentence length |
| **Gold Train Entity Count** | 590 (Kon) / 690 (Asm) / 519 (Sin) | **30 entity mentions** | **17x–23x lower entity density** |
| **Gold Train Entity Density** | ~0.70 entities / sentence (1 every 1.4 sentences) | **0.035 entities / sentence** (1 every 28 sentences) | Extreme entity sparsity |
| **Silver Data Volume Generated** | 7,200 sentences per language ($94k$–$124k$ tokens) | **0 sentences generated** | Exhaustion of raw source entities |
| **Unannotated Corpus Remainder** | Abundant (>50,000 sentences remaining) | **2,751 sentences** ($21,926$ tokens) | Limited total corpus |
| **Entities in Remainder** | Thousands of extractable gazetteer matches | **1 match across 21,926 tokens** (0.04% match rate) | 99.995% Outside (`O`) tokens |

---

## 3. Empirical Diagnosis: Why Weak-Supervision Silver Expansion Fails for Kokborok

1. **Entity Exhaustion in the Source Files**:
   - The 6 source `.xlsx` files provided for Kokborok comprise parallel conversational pairs, sentiment data, and short Q&A dialogues.
   - During Phase 11's stratified annotation, virtually every candidate entity mention across the entire 4,020-sentence collection was identified and stratified into the gold splits (Train: 30, Dev: 17, Test: 16).
   - Scanning the remaining 2,751 unannotated sentences against the validated Kokborok gazetteers yields **exactly 1 entity match** (`LOC: 1`, `PER: 0`, `ORG: 0`, `MISC: 0`) across 21,926 tokens.

2. **The "Outside-Drowning" Mechanism (Evidence from Option b)**:
   - In Phase 12's Option (b) standalone benchmark, an unconstrained model collapsed to **0.00% Dev F1** because 99.55% of tokens in train were `O`.
   - Adding 2,751 weakly-supervised sentences from the remainder would inject **21,925 Outside tokens and only 1 entity token**.
   - This would dilute the entity token ratio from 0.45% down to **0.10%**, severely overwhelming the linear-chain CRF transition matrix with $O \to O$ dominance and extinguishing emission scores for named entities.

3. **Weak Supervision Asymmetry**:
   - In Phase 11 (`reports/kokborok_weak_supervision_precision.md`), gazetteer precision on held-out gold-dev was high (91.67% overall, 100% LOC), but recall was only 64.71% on LOC and **0.00% on PER and ORG**.
   - Propagating gazetteer rules to the remainder would yield 0 PER and 0 ORG mentions, worsening the already severe class imbalance against persons and organizations.

---

## 4. Final Scientific Decision

- **Pipeline Decision**: **NO weak-supervision silver generation or unconstrained pseudo-labeling is applied to Kokborok**.
- **Execution Architecture**: Final training executes strictly under **Option (c) Staged / Protected Modular Approach** on `data/gold/kokborok/train.conll` ($N=850$ sentences, $30$ entities).
- **Hyperparameter Adaptation**:
  - Because Kokborok's gold entity count is sparse ($30$ mentions), small batch sizes ($B=16$) and elevated learning rates ($2\times 10^{-3}$) on the trainable projection head and embeddings provide sufficient gradient magnitude without overfitting or collapsing to the Outside majority class.
  - Frozen multilingual BiLSTM and Self-Attention backbones provide rich contextual representations that protect against the representation starvation that doomed Option (b).
