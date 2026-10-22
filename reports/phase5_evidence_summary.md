# Phase 5 Evidence Summary: Compiled Root Causes (Phases 1–4)

> **Phase 5 Mandatory First-Step Deliverable — Audit Trail & Synthesis**  
> Synthesizes the empirical findings of Phase 1 (Error Taxonomy), Phase 2 (Long-Sentence & Multi-Entity Diagnostic),
> Phase 3 (Data Sufficiency & Annotation Quality), and Phase 4 (Architectural Capacity Sweep).

---

## 1. Cross-Phase Diagnostic Synthesis

### 1.1 Phase 1 — Root-Cause Error Taxonomy Summary
- **Primary Error Modes**: Missed Entities (False Negatives: 34 Konkani, 28 Assamese, 26 Sinhala) and Complete-Sentence Collapse (55 Konkani, 48 Assamese, 38 Sinhala).
- **Secondary Error Modes**: False Positives (concentrated in LOC/toponymic heuristic matches).
- **Negligible Errors**: Boundary under/over-extraction and B-/I- transitions accounted for $<2\%$ of errors, proving that segmentation logic is sound; the failure is full recognition suppression.

### 1.2 Phase 2 — Long-Sentence & Multi-Entity Diagnostic Summary
- **Length vs. Density Correlation**: Sentence length is the primary monotonic driver of collapse ($>34$ tokens: collapse rate jumps from $\approx 0–4\%$ to $25\%–44\%$).
- **72 Collapse Cases Root-Cause Triage**:
  - **OOV-Driven**: $60 / 72$ ($83.3\%$) — cumulative out-of-vocabulary rate across long token sequences.
  - **Orthography-Driven**: $10 / 72$ ($13.9\%$) — script-specific encoding/conjunct irregularities:
    - Sinhala: Missing Zero-Width Joiner (ZWJ, `\u200D`) in conjuncts (e.g. `ප්ර` vs `ප්‍ර`).
    - Assamese: Ra-phala ligature sequence (`\u09CD\u09F0` in `প্ৰচলিত`) and Khanda-Ta (`\u09CE`).
  - **Attention-Degeneracy**: $2 / 72$ ($2.8\%$) — uniform entropy ratio $>0.95$ (Konkani dev `#28` and stress `#223`).
  - **Unexplained**: $0 / 72$ ($0.0\%$).
- **Middle-Entity Suppression**: In sentences with $\ge 3$ entities, middle entities suffer a severe recall penalty in Konkani (recall $33.93\%$ vs $56.00\%$ initial and $72.00\%$ last, a $+30.07\text{ pp}$ position gap).

### 1.3 Phase 3 — Data Sufficiency & Annotation Quality Audit Summary
- **Learning-Curve Slopes**:
  - Konkani: Steeply rising ($\Delta F_1 = +7.09\text{ pp}$ from 75% to 100%, slope $+3.52\text{ pp}/1\text{k sents}$) — highly volume-sensitive.
  - Assamese: Modestly climbing ($\Delta F_1 = +1.70\text{ pp}$, slope $+0.84\text{ pp}/1\text{k sents}$) — early plateau.
  - Sinhala: Steeply rising ($\Delta F_1 = +14.69\text{ pp}$, slope $+7.30\text{ pp}/1\text{k sents}$) — severe data hunger.
- **Frequency Cross-Check**: $57.7\%–84.1\%$ of failure entities appear $>5$ times in the training pool (e.g. `অসম` appears 1,594 times, `බෞද්ධ` appears 322 times, `उत्सव` appears 123 times). Failures are NOT missing dictionary entries; they fail in complex contexts and inflected forms.
- **Silver Quality**: Independent gold-train precision is $78.26\%–85.01\%$. LOC precision is systematically lowest ($57.81\%–77.25\%$) due to toponymic heuristic over-generation.

### 1.4 Phase 4 — Architectural Capacity Sweep Summary
- Tested 4 configurations (control, 2 attention blocks, 8 attention heads, 3 BiLSTM layers).
- Verdict: **Architectural capacity is NOT the bottleneck**. 2 blocks and 3 layers degraded standard Gold-Dev F1 ($-4$ to $-18\text{ pp}$) and failed to resolve sentence collapse or sharpen attention entropy. Current $\sim 10.6\text{M}$ parameter architecture (`1b / 4h / 2l`) is preserved.

---

## 2. Top Evidenced Root Causes per Language

### 2.1 Konkani (`gom`)
1. **Evidenced Root Cause K1: Middle-Entity Recall Collapse in Multi-Entity Sentences**
   - *Evidence*: Phase 2 position analysis documented middle-entity recall collapsing to **33.93%** (vs 56.0% initial and 72.0% last, gap $+30.07\text{ pp}$). CRF linear-chain emission bias heavily penalizes consecutive entity transitions across middle tokens.
   - *Target Metric*: Middle-Entity Recall on stress set and stress F1.
2. **Evidenced Root Cause K2: Locative Case-Marker OOV on Geographic Stems**
   - *Evidence*: Phase 1 error analysis and manual test reproduction showed `पणजींत` (Panaji-in) and `गोयाच्या` (Goa-of) suffered a 50% F1 penalty and 33.3% recall on dev due to inflected suffix forms missing from word embeddings.
   - *Target Metric*: LOC Recall on Dev and LOC-driven collapse reduction.
3. **Evidenced Root Cause K3: Extreme Data Volume Sensitivity**
   - *Evidence*: Phase 3 learning curve showed a steep $+7.09\text{ pp}$ jump between 75% and 100% data.

### 2.2 Assamese (`as`)
1. **Evidenced Root Cause A1: Inflectional Case-Marker OOV on High-Frequency Named Stems**
   - *Evidence*: Phase 3 frequency cross-check revealed `অসম` appears 1,594 times in training, yet inflected variants `অসমত` (locative, 172 occurrences), `অসমৰ` (genitive, 1,030 occurrences), and novel case inflections fail during sentence evaluation, contributing to Phase 2's 79.2% OOV-driven collapse rate.
   - *Target Metric*: LOC Recall on Dev and Stress-Set Entity F1.
2. **Evidenced Root Cause A2: Orthographic Ligature Sequence Divergence**
   - *Evidence*: Phase 2 triage identified Ra-phala ligature sequences (`\u09CD\u09F0` in `প্ৰচলিত`) and Khanda-Ta (`\u09CE`) causing tokenization mismatches triggering complete-sentence collapse.
   - *Target Metric*: Stress-Set Collapse Rate and Token Accuracy.

### 2.3 Sinhala (`si`)
1. **Evidenced Root Cause S1: Missing Zero-Width Joiner (ZWJ) in Orthographic Conjuncts**
   - *Evidence*: Phase 2 triage proved 27.8% of Sinhala collapses are triggered by decomposed or missing ZWJ (`\u200D`) in conjuncts (e.g. `ප්ර` vs `ප්‍ර`, `ක්ර` vs `ක්‍ර`).
   - *Target Metric*: Stress-Set Collapse Rate and Stress-Set Recall.
2. **Evidenced Root Cause S2: LOC Heuristic Precision Degradation & Toponymic Noise**
   - *Evidence*: Phase 3 silver re-audit revealed LOC precision on independent gold train is only **57.81%** (162 false positives), corrupting spatial representations and causing false positive misclassifications on common nouns (`මාවත`, `ගඟ`).
   - *Target Metric*: LOC Precision on Dev and overall precision/recall balance.

---

## 3. Disjointness Confirmation of Candidate Interventions
- Candidate interventions are strictly distinct and target non-overlapping mechanisms:
  1. `sinhala_conjunct_normalization`: Pure orthographic Unicode normalization (targets S1).
  2. `konkani_middle_weighting`: Loss-function class-weighting and transition rebalancing (targets K1).
  3. `assamese_inflectional_augmentation`: Morphological case-suffix expansion on named stems (targets A1).
  4. `konkani_inflectional_loc`: Locative case-marker augmentation for Konkani places (targets K2).
  5. `sinhala_loc_precision_filtering`: Heuristic silver confidence filtering and homograph pruning (targets S2).
- Zero redundant capacity interventions are considered, as Phase 4 decisively ruled out architectural capacity limits.
