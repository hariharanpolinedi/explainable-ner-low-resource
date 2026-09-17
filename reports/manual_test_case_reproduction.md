# Manual Test-Case Programmatic Reproduction Report

**Date**: 2026-09-17  
**Target Sentence**: `"पणजींत जाल्ल्या एका विशेष परिषदेंत प्रमोद सावंत, श्रीपाद नाईक आनी दिगंबर कामत हांणी गोयाच्या आर्थिक विकासा खातीर..."`  
**Expected Benchmark Entities (5 Core Entities)**:
1. `पणजींत` -> `LOC`
2. `प्रमोद सावंत` -> `PER`
3. `श्रीपाद नाईक` -> `PER` (Middle list entity)
4. `दिगंबर कामत` -> `PER`
5. `गोयाच्या` -> `LOC`

---

## 1. Summary of Model Predictions on Manual Sentence

| Checkpoint | Origin Phase | Core Entities Detected | Detection Recall | LOC Detection | Middle PER (`श्रीपाद नाईक`) | Additional Spans |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `finetuned_konkani.pt` | Phase 9 Fine-Tuned | **2 / 5** (40.0%) | 40.0% | **0/2 (MISSED)** | **MISSED** | None |
| `selftrained_konkani_best.pt` | Phase 10 Self-Trained (Baseline) | **2 / 5** (40.0%) | 40.0% | **0/2 (MISSED)** | **MISSED** | None |
| `patched_konkani.pt` | Phase 10/11 Patch v1 (Rejected) | **2 / 5** (40.0%) | 40.0% | **0/2 (MISSED)** | **MISSED** | None |
| `patched_v2_konkani.pt` | Patch v2 (Rejected) | **2 / 5** (40.0%) | 40.0% | **0/2 (MISSED)** | **MISSED** | None |
| `final_konkani.pt` | Currently Deployed Bundle | **2 / 5** (40.0%) | 40.0% | **0/2 (MISSED)** | **MISSED** | None |

---

## 2. Granular Breakdown by Checkpoint

### `finetuned_konkani.pt` (Phase 9 Fine-Tuned)
- **Target Hits**: 2 / 5 (40.0%)
- **Expected Entity Matches**:
  - `पणजींत` (LOC): **MISSED**
  - `प्रमोद सावंत` (PER): **PASSED**
  - `श्रीपाद नाईक` (PER): **MISSED**
  - `दिगंबर कामत` (PER): **PASSED**
  - `गोयाच्या` (LOC): **MISSED**
- **All Spans Predicted by Model**:
  - `[PER]` "प्रमोद सावंत" (tokens 5:7)
  - `[PER]` "दिगंबर कामत" (tokens 11:13)

### `selftrained_konkani_best.pt` (Phase 10 Self-Trained (Baseline))
- **Target Hits**: 2 / 5 (40.0%)
- **Expected Entity Matches**:
  - `पणजींत` (LOC): **MISSED**
  - `प्रमोद सावंत` (PER): **PASSED**
  - `श्रीपाद नाईक` (PER): **MISSED**
  - `दिगंबर कामत` (PER): **PASSED**
  - `गोयाच्या` (LOC): **MISSED**
- **All Spans Predicted by Model**:
  - `[PER]` "प्रमोद सावंत" (tokens 5:7)
  - `[PER]` "दिगंबर कामत" (tokens 11:13)

### `patched_konkani.pt` (Phase 10/11 Patch v1 (Rejected))
- **Target Hits**: 2 / 5 (40.0%)
- **Expected Entity Matches**:
  - `पणजींत` (LOC): **MISSED**
  - `प्रमोद सावंत` (PER): **PASSED**
  - `श्रीपाद नाईक` (PER): **MISSED**
  - `दिगंबर कामत` (PER): **PASSED**
  - `गोयाच्या` (LOC): **MISSED**
- **All Spans Predicted by Model**:
  - `[PER]` "प्रमोद सावंत" (tokens 5:7)
  - `[PER]` "दिगंबर कामत" (tokens 11:13)

### `patched_v2_konkani.pt` (Patch v2 (Rejected))
- **Target Hits**: 2 / 5 (40.0%)
- **Expected Entity Matches**:
  - `पणजींत` (LOC): **MISSED**
  - `प्रमोद सावंत` (PER): **PASSED**
  - `श्रीपाद नाईक` (PER): **MISSED**
  - `दिगंबर कामत` (PER): **PASSED**
  - `गोयाच्या` (LOC): **MISSED**
- **All Spans Predicted by Model**:
  - `[PER]` "प्रमोद सावंत" (tokens 5:7)
  - `[PER]` "दिगंबर कामत" (tokens 11:13)

### `final_konkani.pt` (Currently Deployed Bundle)
- **Target Hits**: 2 / 5 (40.0%)
- **Expected Entity Matches**:
  - `पणजींत` (LOC): **MISSED**
  - `प्रमोद सावंत` (PER): **PASSED**
  - `श्रीपाद नाईक` (PER): **MISSED**
  - `दिगंबर कामत` (PER): **PASSED**
  - `गोयाच्या` (LOC): **MISSED**
- **All Spans Predicted by Model**:
  - `[PER]` "प्रमोद सावंत" (tokens 5:7)
  - `[PER]` "दिगंबर कामत" (tokens 11:13)

---

## 3. Findings: Representative vs Outlier

1. **Representative Failure, Not an Outlier**:
   - The failure of the currently deployed model (`final_konkani.pt` / `selftrained_konkani_best.pt`) to detect `पणजींत` and `गोयाच्या` matches exactly what our LOC diagnostic uncovered: inflected/OOV location names suffer a **50% F1 penalty** on Gold-Dev, with recall collapsing to **33.3%**.
   - The missing middle PER (`श्रीपाद नाईक`) matches the diagnosed middle-entity list resolution weakness in long sentences.
2. **Patch Checkpoint Comparison**:
   - `patched_konkani.pt` and `patched_v2_konkani.pt` also failed to recognize `पणजींत` and `गोयाच्या`, proving that neither patch ever addressed the underlying morphological case-inflection cause of LOC failure.