# Root-Cause Error Taxonomy Census: Konkani

- **Language**: Konkani
- **Model Evaluated**: `baseline_v1/checkpoints/final_konkani.pt` (Frozen Phase 0 Baseline)
- **Gold Diagnostic Population**: 247 sentences, 5061 tokens, 251 gold entities
- **Total Identified Errors**: 90

---

## 1. Ranked Error Taxonomy Table

| Rank | Error Category | Raw Count | % of Total Errors | % of Gold Entities | Statistical Status |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | **Complete Sentence Collapse** (`complete_sentence_collapse`) | **55 / 90** | 61.11% | 21.91% | **Robust (N >= 30)** |
| 2 | **Missed Entity** (`missed_entity`) | **34 / 90** | 37.78% | 13.55% | **Robust (N >= 30)** |
| 3 | **False Positive** (`false_positive`) | **1 / 90** | 1.11% | 0.40% | **[Statistically Fragile (N < 10)]** |
| 4 | **Wrong Entity Type** (`wrong_entity_type`) | **0 / 90** | 0.00% | 0.00% | **[Statistically Fragile (N < 10)]** |
| 5 | **Incomplete Span** (`incomplete_span`) | **0 / 90** | 0.00% | 0.00% | **[Statistically Fragile (N < 10)]** |
| 6 | **Overextended Span** (`overextended_span`) | **0 / 90** | 0.00% | 0.00% | **[Statistically Fragile (N < 10)]** |
| 7 | **Bi Transition Error** (`bi_transition_error`) | **0 / 90** | 0.00% | 0.00% | **[Statistically Fragile (N < 10)]** |
| 8 | **O Inside Entity** (`o_inside_entity`) | **0 / 90** | 0.00% | 0.00% | **[Statistically Fragile (N < 10)]** |

---

## 2. Held-Out Raw Corpus Diagnostic Census (Candidate Anomaly Quantification)

Predictions on unannotated raw sentences from the corpus pool (`data/processed/sentences.txt`),
screened for zero overlap against `test.conll` and used to quantify collapse and structural tag consistency in the wild:

- **Raw Sample Size**: 1500 sentences (17984 tokens)
- **Zero-Entity Predicted Sentences (Pure O)**: 1441 / 1500 (96.07%)
- **Sentences with Known Gazetteer Entities**: 247
- **Gazetteer Sentences with Complete Model Collapse**: 192 / 247 (77.73%)
- **Structural B-/I- Transition Errors in the Wild**: 0 (Zero violations; CRF Viterbi decoding maintains strict state machine validity)

---

## 3. Key Findings & Diagnostic Synthesis

1. **Dominant Failure Mode**: `complete_sentence_collapse` accounts for **55 / 90 (61.11%)** of all observed errors.
2. **Sentence Collapse Severity**: Complete-sentence collapse represents a severe vulnerability where the model outputs all-O despite valid entities being present.
3. **Absence of Transition Defects**: `bi_transition_error` is at 0 raw count across both gold and raw corpora, confirming the CRF transition parameterization is mathematically sound.
4. **Intervention Guidance for Phase 5**: Reliability engineering must prioritize sentence collapse recovery and lexical/inflectional false negative suppression.
