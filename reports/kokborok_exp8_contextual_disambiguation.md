# Kokborok Experiment 8 — Contextual LOC-vs-ORG Disambiguation Report

## 1. Objective

The objective of **Experiment 8** is to test whether targeted contextual disambiguation training can improve Kokborok entity-level NER F1, specifically resolving boundary and type confusion between ambiguous locations (`LOC`) and multi-token organizations (`ORG`).

---

## 2. Research Hypothesis

**Hypothesis**: A proper noun (e.g., *Tripura*, *Agartala*) is predominantly a Location (`LOC`) when standing alone or attached to spatial/case markers (*Tripura o*, *Tripura ni*), but transitions to an Organization (`ORG`) when accompanied by institutional head nouns (e.g., *University*, *Board*, *Department*, *Police*, *Channel*, *Bank*, *Corporation*). We test whether explicitly exposing the model to paired contextual patterns improves general entity-level disambiguation on Gold-Dev without degrading existing high-precision boundaries.

---

## 3. Data Sources

The following gold and clean public datasets were audited and utilized:
1. **Gold Train**: [`data/gold/kokborok/train.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/kokborok/train.conll) (850 sentences).
2. **Gold Dev**: [`data/gold/kokborok/dev.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/gold/kokborok/dev.conll) (200 sentences).
3. **Cleaned Public ORG Data**: [`data/processed/kokborok_org_aug_clean.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_org_aug_clean.conll) (57 clean public sentences from Experiment 7).
4. **Precision Baseline Pool**: [`data/processed/kokborok_exp3_precision_train.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_exp3_precision_train.conll).

---

## 4. Initial Train / Dev Audit

A systematic pre-training audit of `Tripura` and institutional terms in Gold Train and Gold Dev revealed:

| Lexical Token | Dataset | Occurrences | Entity Tags Assigned | Syntactic / Semantic Context |
| :--- | :--- | :--- | :--- | :--- |
| **`Tripura`** | Gold Train | 5 | `B-ORG` (3), `B-LOC` (2) | `B-LOC` in geographical frames (*Tripura o*); `B-ORG` in institutional names (*Tripura channel*, state body). |
| **`Tripura`** | Gold Dev | 3 | `B-LOC` (2), `B-ORG` (1) | `B-LOC` in locative phrases (*Tripura o*, *Tripura ni*); `B-ORG` in bulletin title (*Tripura Hukbar Bedek*). |
| **`Agartala`** | Gold Train | 12 | `B-LOC` (12) | Capital city location (*Agartala o*, *Agartala ni*). |
| **`Agartala`** | Gold Dev | 4 | `B-LOC` (2), `O` (2 unannotated) | Capital city location. |

---

## 5. Ambiguity & Context Matrix

| Lexical Context Pattern | Typical Entity Class | Gold Train Count | Gold Dev Count | Exemplar Pattern | Ambiguity Level |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `<Toponym>` + Postposition (*o / ni / haste*) | **LOC** | 14 | 4 | *Tripura o*, *Agartala ni* | Low |
| `<Toponym>` + `University` | **ORG** | 27 | 0 | *Tripura University*, *ICFAI University* | Low |
| `<Toponym>` + `Board` / `board` | **ORG** | 24 | 0 | *Tripura board*, *Tripura Board* | Moderate (casing sensitive) |
| `<Toponym>` + `Police` / `police` | **ORG** | 30 | 0 | *Tripura Police*, *Delhi police* | Low |
| `<Toponym>` + `Bank` | **ORG** | 36 | 0 | *Tripura Gramin Bank*, *SBI* | Low |
| `<Toponym>` + `Bedek` / `Hukbar` | **ORG** | 42 | 1 | *Tripura Hukbar Bedek*, *Twimung Bedek* | Low |
| `<Toponym>` + `Corporation Ltd.` | **ORG** | 1 | 0 | *Tripura State Electricity Corporation Ltd.* | Low |
| Standalone Common Nouns (*department, bank, school*) | **O** | 225+ | 5 | *ani department o*, *bank account* | High (hard negatives) |

---

## 6. Data Leakage & Integrity Audit

A strict programmatic audit confirmed:
```text
Exact Overlap with Gold Test (data/gold/kokborok/test.conll):  0 (0.00% — STRICT ZERO LEAKAGE)
Exact Overlap with Gold Dev  (data/gold/kokborok/dev.conll):   0 (0.00% — ZERO OVERLAP)
Exact Overlap with Gold Train(data/gold/kokborok/train.conll): 0 (Filtered)
```
The locked test set remained completely untouched throughout dataset construction and experimentation.

---

## 7. Exact Augmentation Dataset Construction

We constructed the Experiment 8 training pool by:
1. Taking the clean baseline training data.
2. Integrating the 57 verified contextual public ORG sentences.
3. Filtering all exact overlaps with Gold-Dev and Gold-Test.
4. Output file: [`data/processed/kokborok_exp8_context_train.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_exp8_context_train.conll).

---

## 8. Number of Sentences

- **Total Training Sentences in Exp 8 Pool**: **5,045 sentences**
- **Gold-Dev Validation Set**: **200 sentences** (17 gold entities)

---

## 9. Number of Entities by Type in Training Pool

| Entity Type | Training Entity Count | Proportion |
| :--- | :--- | :--- |
| **LOC** | 1,202 | 41.0% |
| **ORG** | 967 | 33.0% |
| **PER** | 602 | 20.5% |
| **MISC** | 162 | 5.5% |
| **TOTAL** | **2,933** | **100.0%** |

---

## 10. Model Architecture

- **Backbone**: 4-Language Multilingual NER Architecture ([`model/ner_model_v4_multilingual.py`](file:///c:/Users/harih/Desktop/NLP-MARK-3/model/ner_model_v4_multilingual.py)).
- **Modules**: Shared BiLSTM Encoder (2-layer, 128 hidden dim) + Shared Multi-Head Self-Attention + Language Conditioning + Per-Language Linear-CRF Heads.

---

## 11. Frozen vs. Trainable Parameters

- **Frozen Parameters**: 1,012,621 parameters (Shared BiLSTM, Attention, Language Embedding, Konkani/Assamese/Sinhala Heads).
- **Trainable Parameters**: 10,200,702 parameters (Kokborok Head `fc_heads[3]`, Kokborok CRF `crf_heads[3]`, Word Embeddings, Char-CNN).

---

## 12. Hyperparameters

- **Optimizer**: `AdamW` ($\text{weight\_decay} = 10^{-4}$)
- **Learning Rates**: $\text{lr}_{\text{head}} = 5\times 10^{-5}, \text{lr}_{\text{emb}} = 3\times 10^{-5}, \text{lr}_{\text{crf}} = 2\times 10^{-5}$
- **Gradient Clipping**: $5.0$
- **Batch Size**: 32
- **Epochs**: 6

---

## 13. GPU & Device Information

- **Device**: `cuda:0`
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU
- **CUDA Capability**: 8.6

---

## 14. Random Seeds

Deterministic runs were conducted across **3 fixed random seeds**: `42`, `123`, `2026`.

---

## 15. Per-Seed Gold-Dev Results

| Seed | Initial Dev F1 | Best Epoch | Best Dev F1 | Dev Precision | Dev Recall | Dev TP | Dev FP | Dev FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Seed 42** | 94.44% | 0 | **94.44%** | 89.47% | 100.00% | 17 | 2 | 0 |
| **Seed 123** | 94.44% | 0 | **94.44%** | 89.47% | 100.00% | 17 | 2 | 0 |
| **Seed 2026** | 94.44% | 0 | **94.44%** | 89.47% | 100.00% | 17 | 2 | 0 |

---

## 16. Multi-Seed Statistical Summary (Mean ± Std)

- **Overall Dev Strict F1**: $\mathbf{94.44\% \pm 0.00\%}$
- **Min Dev F1**: $94.44\%$
- **Max Dev F1**: $94.44\%$
- **Mean Precision**: $89.47\%$
- **Mean Recall**: $100.00\%$

---

## 17. Gold-Dev Error Matrix

| Entity Type | Gold Mentions | Predicted Mentions | TP | FP | FN | Precision | Recall | Strict F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PER** | 2 | 2 | 2 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| **LOC** | 9 | 11 | 9 | 2 | 0 | 81.82% | 100.00% | **90.00%** |
| **ORG** | 2 | 2 | 2 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| **MISC** | 4 | 4 | 4 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| **TOTAL** | **17** | **19** | **17** | **2** | **0** | **89.47%** | **100.00%** | **94.44%** |

---

## 18. Per-Type Performance (PER / LOC / ORG / MISC)

- **PER**: $\text{F1} = \mathbf{100.00\%}$ (2/2 TP, 0 FP, 0 FN)
- **LOC**: $\text{F1} = \mathbf{90.00\%}$ (9/9 TP, 2 FP on unannotated lowercase *agartala*, 0 FN)
- **ORG**: $\text{F1} = \mathbf{100.00\%}$ (2/2 TP, 0 FP, 0 FN)
- **MISC**: $\text{F1} = \mathbf{100.00\%}$ (4/4 TP, 0 FP, 0 FN)

---

## 19. Comparison Against Experiment 7

| Experiment | Configuration | Dev Precision | Dev Recall | Dev F1 | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 4 Baseline** | Optimal Precision Tuning | 89.47% | 100.00% | **94.44%** | Authoritative Frozen Model |
| **Exp 7** | Raw ORG Data Augmentation (0.5x / 1.0x) | 89.47% | 100.00% | **94.44%** | No Dev Gain Over Exp 4 |
| **Exp 8** | Contextual LOC-vs-ORG Disambiguation Pool | 89.47% | 100.00% | **94.44%** | No Dev Gain Over Exp 4 |

---

## 20. Decision Gate

```text
NO IMPROVEMENT OVER 94.44% DEV F1 — OFFICIAL TEST EVALUATION IS NOT JUSTIFIED.
```

The experiment evaluated paired contextual data on Gold-Dev. Because the authoritative baseline checkpoint already captures 100% of Gold-Dev entities ($\text{Recall} = 100.0\%$, $\text{Dev F1} = 94.44\%$) and further parameter updates on the training pool do not improve Dev precision without recall degradation, the gate strictly forbids test set evaluation.

---

## 21. Locked Test Evaluation Status

**NOT EVALUATED (Sealed)**. In strict adherence to scientific integrity constraints, zero locked-test evaluations were performed.

---

## 22. Scientific Interpretation

1. The experiment confirms that the current model [`checkpoints/kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt) has reached a high performance ceiling on Gold-Dev ($\text{Recall} = 100.00\%$, $\text{ORG F1} = 100.00\%$, $\text{Overall F1} = 94.44\%$).
2. The residual 2 false positives on Gold-Dev are caused by unannotated lowercase `agartala` tokens in the reference data, rather than model hallucination.
3. Additional fine-tuning epochs on augmented corpora lead to representation drift on the small Gold-Dev distribution; hence, keeping the converged `kokborok_exp4_best.pt` checkpoint is optimal.

---

## 23. Limitations

- The Kokborok Gold-Dev set contains $N=17$ gold entities. With $17/17$ true positives already achieved, Gold-Dev has zero false negatives to recover.
- Model selection cannot rely on locked-test feedback; therefore, maintaining frozen checkpoints prevents overfitting.

---

## 24. Reproducibility Commands

```powershell
# 1. Build zero-leakage contextual training pool
python training/build_exp8_pool.py

# 2. Run multi-seed training and Gold-Dev evaluation
python training/train_kokborok_exp8_context.py
```

---

## 25. Final Recommendation & Classification

### Final Classification:
```text
FAILURE (To improve beyond authoritative Gold-Dev baseline)
```

### Final Recommendation:
- Preserve **[`checkpoints/kokborok_exp4_best.pt`](file:///c:/Users/harih/Desktop/NLP-MARK-3/checkpoints/kokborok_exp4_best.pt)** as the authoritative frozen model (**82.35% Official Locked Test F1**, **94.44% Dev F1**).
- Do not deploy or promote Experiment 8 checkpoints.
