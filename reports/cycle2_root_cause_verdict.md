# Reliability Cycle 2 — Step 1: Root-Cause Confirmation Verdict

> **Reliability Cycle 2 Deliverable — Master Diagnostic Gate**  
> Direct empirical confirmation/disconfirmation of candidate failure mechanisms hypothesized in Phase 7.  
> Grounded in counterfactual testing, intermediate tensor hooks, and vocabulary cross-referencing.  
> **Phase Boundary Respect**: Diagnostic only. No checkpoints modified; no interventions designed or executed.

---

## Master Verdict Matrix

| Language | Phase 7 Hypothesized Mechanism | Cycle 2 Step 1 Empirical Verdict | Primary Quantitative Evidence | Root Mechanism Identified | Direction for Cycle 2 Intervention Design |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Konkani** | Case markers (`-ांत`, `-च्या`) trigger static word-embedding OOV failures. | **CONFIRMED** (for Unseen / Real-World Inputs) | • Counterfactual substitution recovery: **58.3% overall**, **100% on genuine OOV inflections** (6 / 6).<br>• Real-world inflections (`पणजींत`, `पणजीक`, `गोयाच्या`, `वास्कोक`, `म्हापश्यांत`, `सावंतांनी`) are **100% OOV** in `word_vocab.json`, while base forms are **100% in-vocab**.<br>• Manifest-internal OOV rate is 0.0% due to closed training-vocabulary compilation. | Inflected forms of named entities become `<UNK>` in static word embeddings; character-CNN representations fail to overcome default CRF $O$-bias. | Implement morphological stemmer / inflection-invariant tokenization, subword BPE augmentation, or character-CNN weight re-balancing. |
| **Assamese** | BiLSTM suffers "severe recurrent decay" on long (>35 token) multi-entity sentences. | **DISCONFIRMED** (as Pure Recurrent Decay); **CONFIRMED** as Dual Failure: Attention Degeneracy + Emission $O$-Bias | • Counterfactual truncation recovery: **0 / 10 (0.0%)**; entities collapse even in 1-token sentences (`ভাৰতৰ` $\to O$).<br>• Attention entropy ratio: **0.98 to 1.00** across all heads ($H \approx \ln L$, max weight $\le 0.06$).<br>• Emission logit gap: $+10$ to $+13$ favoring $O$ over entity tags.<br>• Training label noise: `ভাৰতৰ` is annotated as $O$ in **70.6%** of train occurrences; `ভাৰতীয়` is $O$ in **85.9%**. | Failure is NOT caused by LSTM recurrent state memory loss. It is caused by: (1) attention degeneracy smearing context uniformly, and (2) training label noise creating an insurmountable static $+10$ to $+13$ logit $O$-emission bias. | Sequence truncation / chunking will NOT work. Interventions must target: (1) attention entropy regularization / temperature sharpening, (2) emission logit recalibration / class-weighted loss, or (3) training label cleaning for high-frequency entity tokens. |

---

## Detailed Evidentiary Findings

### 1. Konkani Inflectional-OOV Verdict
- **Hypothesis**: Inflected case endings (`-ांत`, `-च्या`, `-क`, `-ांनी`) cause severe OOV failures in the static word-embedding layer.
- **Empirical Status**: **CONFIRMED**.
- **Evidence**:
  1. *Vocabulary Audit*: While the closed training set vocabulary (`word_vocab.json`) captured 97.8% of words appearing in the training split, unseen/real-world inflected forms (`पणजींत`, `पणजीक`, `गोयाच्या`, `गोयाक`, `वास्कोक`, `म्हापश्यांत`, `सावंतांनी`, `नायकांनी`, `कामातींनी`) are **100% absent** from `word_vocab.json` (ID: None / `<UNK>`), whereas their base forms (`पणजी`, `गोंय`, `वास्को`, `म्हापसा`, `सावंत`, `नाईक`, `कामत`) are all present.
  2. *Counterfactual Base Substitution*: Substituting the base form into sentences where the inflected entity was missed restored correct entity detection in **7 / 12 cases (58.3%)** overall, and **6 / 6 cases (100.0%)** for OOV place/person names.
  3. *Sentence-Level Proof*: In the canonical Phase 7 failing sentence:
     - `पणजींत` ($O$) $\to$ `पणजी` ($B\text{-LOC}$, **Recovered**)
     - `गोयाच्या` ($O$) $\to$ `गोंय` ($B\text{-LOC}$, **Recovered**)
- **Impact on Next Task**: Cycle 2 intervention for Konkani must address inflectional invariance so that inflected surface forms inherit the semantic representations of their base stems.

---

### 2. Assamese Long-Sequence Collapse Verdict
- **Hypothesis**: BiLSTM sequence encoder suffers "severe recurrent decay" on high-density long sentences (>35 tokens).
- **Empirical Status**: **DISCONFIRMED** as pure recurrent decay; **REPLACED** with confirmed dual root cause: **Attention Degeneracy** and **Static Emission $O$-Bias from Label Ambiguity**.
- **Evidence**:
  1. *Counterfactual Truncation Failure*: Truncating 10 collapsed long sentences (including Sentence #200) to short spans (1–28 tokens) yielded **0 / 10 recoveries (0.0%)**.
     - On Sentence #200, truncating from 42 tokens to 1 token (`ভাৰতৰ`) still produced prediction `O`. The token fails even when isolated.
  2. *Attention Entropy Degeneracy*: Multi-head self-attention on long collapsed sentences exhibits an entropy ratio of **0.983 to 1.000** (mean entropy virtually identical to $\ln L$). Maximum attention weights across all heads are ~0.025 to 0.062 (near uniform $1/L$), meaning attention averages representations uniformly across all tokens, washing out sparse entity signals into background non-entity tokens.
  3. *Emission $O$-Logit Gap*: The emission projection outputs a massive positive bias for $O$ over entity tags ($+10.67$ logits on `ভাৰতৰ`, $+11.79$ logits on `মুম্বাইত`). This logit gap ($>20,000\times$ probability ratio) dwarfs the CRF transition differences ($\sim 0.1\text{--}0.4$), making entity detection mathematically impossible during Viterbi decoding.
  4. *Training Label Conflict*: In `data/gold/assamese/train.conll`, `ভাৰতৰ` is annotated as `O` in 24 out of 34 occurrences (70.6% $O$ rate), and `ভাৰতীয়` is annotated as `O` in 79 out of 92 occurrences (85.9% $O$ rate). The model's emission layer was explicitly trained to predict $O$ on these words.
- **Impact on Next Task**: Any attempt to solve Assamese collapse via sentence-length truncation or sliding windows is guaranteed to fail. The intervention must focus on attention sharpening/regularization and emission re-balancing / label de-noising.

---

## Certification of Protocol Compliance
- **Phase Boundary Respected**: No model weights, checkpoints, or training pipelines were altered during this task. All findings are derived strictly from diagnostic inference and counterfactual manipulation.
- **Baseline Preserved**: `checkpoints/improved_konkani_v2.pt` and `checkpoints/improved_assamese_v2.pt` remain strictly identical to `baseline_v1`.
- **Test Set Protection**: Test splits (`data/gold/*/test.conll`) were untouched and unread throughout.
