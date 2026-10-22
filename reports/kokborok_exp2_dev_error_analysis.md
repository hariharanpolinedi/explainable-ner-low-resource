# Kokborok Gold-Dev Detailed Error Analysis Report (Experiment 2, Step 2)

**Evaluated Checkpoint:** `checkpoints/updated_kokborok_optionC.pt`  
**Dataset:** `data/gold/kokborok/dev.conll` (200 sentences, 1,612 tokens, 17 gold entities)  
**Evaluation Protocol:** Strict span-level BIO matching on GPU (`cuda`)  
**Date:** 2026-09-30  

---

## 1. Summary Performance Metrics

| Entity Type | Precision | Recall | Entity F1 | TP | FP | FN | Gold Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PER** | 0.00% | 0.00% | **0.00%** | 0 | 0 | 2 | 2 |
| **LOC** | 75.00% | 33.33% | **46.15%** | 3 | 1 | 6 | 9 |
| **ORG** | 0.00% | 0.00% | **0.00%** | 0 | 0 | 2 | 2 |
| **MISC** | 100.00% | 100.00% | **100.00%** | 4 | 0 | 0 | 4 |
| **OVERALL** | **87.50%** | **41.18%** | **56.00%** | **7** | **1** | **10** | **17** |

---

## 2. Categorization of All Errors (Categories A–I)

### Category A: False Negatives (FN = 10)
1. **Sentence 1 (`PER['kanika']`):** `My name is kanika Nung tamo khwlainai`
   - *Cause:* Lowercase first letter in informal English/Kokborok conversational chat.
2. **Sentence 2 (`PER['Parimal Debbarma']`):** `Parimal Debbarma hwnei korok sa engineer...`
   - *Cause:* Multi-token person name at sentence start.
3. **Sentence 3 (`ORG['Tripura Hukbar Bedek']`):** `Tripura Hukbar Bedek Bwchwlwi magwnang...`
   - *Cause:* Tri-token tribal government department (*Hukbar Bedek* = Agriculture).
4. **Sentence 4 (`ORG['DD news']`):** `Nini mokol kwrwi DD news channelo naidi`
   - *Cause:* Bi-token media name (*Doordarshan News*).
5. **Sentence 5 (`LOC['Lunglei']`):** `Lunglei song advent Christmas khlaina...`
   - *Cause:* Regional district town located in Mizoram across the Tripura border.
6. **Sentence 9 (`LOC['Udaipur']`):** `Udaipur ni peratiya ow bike accidents...`
   - *Cause:* Followed by clitic particle `ni` at sentence start in complex syntax.
7. **Sentence 10 (`LOC['Tripurao']`):** `Belai watwi wama bagwi Tripurao twilom kha .`
   - *Cause:* Fused locative suffix `-o` attached to the noun (`Tripura` + `o`).
8. **Sentence 13 (`LOC['Udaipur']`):** `Koita oh norok Udaipur cross khulainai`
   - *Cause:* Medial sentence position in conversational question.
9. **Sentence 14 (`LOC['Agratalao']`):** `Agratalao Bithi kwrwi muikwthwng...`
   - *Cause:* Dual error: Human spelling typo in gold set (`Agratala` instead of `Agartala`) + fused locative `-o`.
10. **Sentence 15 (`LOC['Kailashahar-o']`):** `Kailashahar-o salba niyagolo curfew...`
    - *Cause:* Hyphenated locative clitic (`-o`).

---

### Category B: False Positives (FP = 1)
1. **Sentence 3 (`LOC['Tripura']`):**
   - Model predicted `LOC['Tripura']` on token 0, whereas gold span was multi-token `ORG['Tripura Hukbar Bedek']` across tokens 0..2.

---

### Category C & D: Wrong Entity Type & Boundary Errors
- **Sentence 3:** `Tripura Hukbar Bedek` (Tokens 0..2).
  - Model truncated boundary to token 0 (`Tripura`) and misclassified type as `LOC` instead of `ORG`.

---

### Category E: Complete Sentence Collapse
- No complete sentence collapse observed on non-entity sentences. All 184 negative-control sentences correctly produced all-`O` tags.

---

### Category F: Multiple-Entity Failures
- **Sentence 5 (`Lunglei` [LOC] + `Christmas` [MISC]):**
  - Model correctly detected `Christmas` (MISC) but missed `Lunglei` (LOC).

---

### Category G: Long-Sentence Failures
- **Sentence 9 (19 tokens) & Sentence 16 (18 tokens):**
  - Sentence 9 missed `Udaipur` in a 19-token long accident description.
  - Sentence 16 correctly detected `Tripura` in an 18-token sentence.

---

### Category H: OOV / Rare-Token Failures
- `kanika`: Common Indian female given name, rare in Kokborok text.
- `Hukbar Bedek`: Kokborok specific compound noun for Agriculture Department.
- `DD`: English loan acronym for Doordarshan.

---

### Category I: Inflected / Fused-Form Failures (3 Critical Failures = 30% of all FNs!)
1. `Tripurao` (Sentence 10): LOC + locative case clitic `-o`.
2. `Agratalao` (Sentence 14): LOC typo + locative case clitic `-o`.
3. `Kailashahar-o` (Sentence 15): LOC + hyphenated locative `-o`.
- **Finding:** 30% of all false negatives in `dev.conll` stem directly from unhandled locative `-o` clitics!
