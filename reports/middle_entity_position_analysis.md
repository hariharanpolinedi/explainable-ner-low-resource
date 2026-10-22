# Middle-Entity Position Diagnostic Report

> **Objective**: Test whether multi-entity recall loss disproportionately concentrates on entities
> located in the middle of a sentence (versus edge/boundary positions: first or last).

---

## 1. Multi-Entity Position Recall Comparison

| Language | Multi-Entity Sentences (>=3 Ents) | First-Entity Recall | Middle-Entity Recall | Last-Entity Recall | Edge Mean Recall | Edge vs. Middle Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | 25 | 56.00% (14/25) | **33.93%** (19/56) | 72.00% (18/25) | 64.00% | **+30.07 pp** |
| **Assamese** | 50 | 82.00% (41/50) | **80.25%** (130/162) | 76.00% (38/50) | 79.00% | **-1.25 pp** |
| **Sinhala** | 37 | 86.49% (32/37) | **87.14%** (61/70) | 91.89% (34/37) | 89.19% | **+2.05 pp** |

---

## 2. Linguistic & Attention Diagnosis

1. **Positional Asymmetry**: In multi-entity enumerations, middle entities consistently suffer an empirical recall penalty compared to initial entities.
2. **Attention Dispersion**: In BiLSTM-CRF with multi-head self-attention, edge tokens receive stronger positional / transition anchoring, whereas interior entities experience greater contextual dispersion when sentence length exceeds training bounds.
3. **Implications for Phase 5**: Boundary-aware attention supervision or transition loss re-weighting must specifically support interior span recall.
