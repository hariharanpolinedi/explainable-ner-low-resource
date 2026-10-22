# Kokborok Fresh-Sentence Generalization Audit Report

**Phase**: Post-Phase 14 Verification / Integrity Audit  
**Auditor**: Senior ML/NLP Research Engineer  
**Date**: 2026-09-30  
**Device**: `NVIDIA GeForce RTX 3050 Laptop GPU` (CUDA 12.8 / PyTorch 2.6.0)  
**Evaluation Scope**: 14 Verified Genuinely Fresh Kokborok Sentences (19 Gold Entities)  
**Status**: **GENERALIZATION DEFICIT CONFIRMED — 0.00% OUT-OF-DOMAIN F1**

---

## 1. Objective & Methodology

To determine whether the reported improvements in Kokborok NER (80.00% in Round 1, claimed 90.32% in Round 4) reflect genuine neural linguistic generalization or merely test-set memorization and template overfitting, a fresh-sentence generalization test was designed following the strict protocol established in Phase 14:
1. **Source**: Authentic open Kokborok text from the public `sdmy/kokborok` corpus.
2. **Non-Overlap Discipline**: Every candidate sentence was cross-referenced against all prior training, validation, testing, and synthetic datasets.
3. **Evaluation**: Conducted on GPU using strict span-level BIO matching across all four candidate model checkpoints.

---

## 2. Rigorous Non-Overlap Verification

The 14 candidate sentences were checked against a corpus of **10,014 unique sentences** compiled from:
* `data/gold/kokborok/train.conll`
* `data/gold/kokborok/dev.conll`
* `data/gold/kokborok/test.conll`
* `data/processed/kokborok_improved_train.conll`
* `data/processed/kokborok_calib_train.conll`
* `data/processed/kokborok_surgical_train.conll`

```
Total known sentences indexed from prior splits: 10,014
Sentence  1: Verified ZERO overlap [PASS]
Sentence  2: Verified ZERO overlap [PASS]
Sentence  3: Verified ZERO overlap [PASS]
Sentence  4: Verified ZERO overlap [PASS]
Sentence  5: Verified ZERO overlap [PASS]
Sentence  6: Verified ZERO overlap [PASS]
Sentence  7: Verified ZERO overlap [PASS]
Sentence  8: Verified ZERO overlap [PASS]
Sentence  9: Verified ZERO overlap [PASS]
Sentence 10: Verified ZERO overlap [PASS]
Sentence 11: Verified ZERO overlap [PASS]
Sentence 12: Verified ZERO overlap [PASS]
Sentence 13: Verified ZERO overlap [PASS]
Sentence 14: Verified ZERO overlap [PASS]

VERDICT: Strict 100% Zero-Overlap Confirmed across all 14 sentences.
```

---

## 3. Fresh Sentence Manifest & Gold Annotations

The 14 sentences contain a total of **19 named entities** and 3 negative control sentences ($N=0$ entities):

| # | Kokborok Sentence Text | Gold Named Entities | Category |
|---|---|---|:---:|
| **1** | `Bini achukthaino bini bwsla Rajiv Gandhi naharkha .` | `Rajiv Gandhi` (PER) | PER |
| **2** | `Salsa , Ram kami twih himwi thang tongphuru , chwlarok bodol kaisano nugo .` | `Ram` (PER) | PER |
| **3** | `Indira Gandhi puila bwrwirokni bisingo baraot hani montri bokhorok chalangnai .` | `Indira Gandhi` (PER) | PER |
| **4** | `Bini bupha Jawaharlal Nehru , bono Barot hani puila Montri Bokhorok .` | `Jawaharlal Nehru` (PER), `Barot` (LOC) | PER, LOC |
| **5** | `Indira Gandhi Barot tei England hao rwngswrwngkha tei Feroze Gandhi bai kaijakha .` | `Indira Gandhi` (PER), `Barot` (LOC), `England` (LOC), `Feroze Gandhi` (PER) | PER, LOC |
| **6** | `Bohrokni khoroknwi bwsa chwla wngjakha , Rajiv tei Sanjay .` | `Rajiv` (PER), `Sanjay` (PER) | PER |
| **7** | `Oh tangbwtangrok paithago abo Indian Supreme Court bai rosai naharjakha .` | `Indian Supreme Court` (ORG) | ORG |
| **8** | `Navratri ni salchuku o , Parvati mungwi khoroksa bwrwi sikli hai hwnwi sajakha .` | `Navratri` (MISC), `Parvati` (PER) | MISC, PER |
| **9** | `Kamini lukurok belai khaphurjakha Shiva bohrokno mwthangmani bagwi .` | `Shiva` (PER) | PER |
| **10** | `Malgudi mungwi kaisa chikonsa aoli o , Ramasamy mungwi khoroksa Police inspector tongmani .` | `Malgudi` (LOC), `Ramasamy` (PER) | LOC, PER |
| **11** | `Salsa , Krishnan mungwi khoroksa chwla sikla no bono Ramasamy romkha .` | `Krishnan` (PER), `Ramasamy` (PER) | PER |
| **12** | `Horhapung ni phatimungrok ha bisingo belai swkwlwkmung khlaima bagwi se wngu .` | *None* | Negative Control |
| **13** | `Dal makhani song baikha hwnkhe , bohrok chalaina bagwi achuklaikha .` | *None* | Negative Control |
| **14** | `Bini mokol kotor kotor tei kirima gwnang wngjak .` | *None* | Negative Control |

**Gold Entity Summary**: Total $N=19$ (PER: 13, LOC: 4, ORG: 1, MISC: 1).

---

## 4. Multi-Checkpoint Comparative Evaluation

All candidate Kokborok checkpoints were evaluated on GPU against the 14 fresh sentences:

| Model Checkpoint | Entity Precision | Entity Recall | **Entity F1** | Token Accuracy | TP | FP | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Phase 13 Baseline (`staged_optionC_kokborok.pt`)** | 0.00% | 0.00% | **0.00%** | 83.75% | 0 | 0 | 19 |
| **Round 1 Model (`kokborok_improved_v2.pt`)** | 0.00% | 0.00% | **0.00%** | 83.13% | 0 | 3 | 19 |
| **Round 4 Model (`kokborok_calib_best.pt`)** | 0.00% | 0.00% | **0.00%** | 83.75% | 0 | 0 | 19 |
| **Final Deployed Model (`final_kokborok.pt`)** | 0.00% | 0.00% | **0.00%** | 83.75% | 0 | 0 | 19 |

---

## 5. Token-Level Prediction Error Analysis

Examining the raw predictions from `kokborok_improved_v2.pt` illustrates the exact nature of the failure:

1. **Sentence 1**: Gold: `Rajiv Gandhi` (PER)  
   * Predicted: `bwsla` (B-MISC), `Rajiv` (I-MISC), `Gandhi` (I-MISC)  
   * *Diagnosis*: Misclassified as MISC and included preceding common noun `bwsla` ("son") inside the entity span.
2. **Sentence 3**: Gold: `Indira Gandhi` (PER)  
   * Predicted: `Indira` (B-LOC), `Gandhi` (O)  
   * *Diagnosis*: Falsely classified as LOC due to strong place prior; dropped surname.
3. **Sentence 7**: Gold: `Indian Supreme Court` (ORG)  
   * Predicted: `Supreme` (B-MISC), `Court` (I-MISC)  
   * *Diagnosis*: Dropped `Indian` from entity boundary; misclassified as MISC.
4. **Sentences 2, 4, 5, 6, 8, 9, 10, 11**: All entities completely missed (tagged as `O`).
5. **Sentences 12, 13, 14 (Negative Controls)**: Correctly predicted as `O` across all tokens (yielding 83.75% token accuracy).

---

## 6. Scientific Conclusion

* The model has learned high precision on negative non-entity text (`O` tags), but **zero genuine entity boundary and typing generalization on unseen out-of-domain Kokborok text**.
* The high scores observed on `test.conll` (80.00% in Round 1, 90.32% in Round 4) were strictly driven by memorizing the specific 16 entities present in that 200-sentence split (`Mohan`, `Rahul`, `Tripura`, `Agartala`, `Udaipur`, `Tripura board`, `Christmas`).
* **Conclusion**: Kokborok cannot be scientifically claimed to have achieved a 90%+ generalizing NER system.
