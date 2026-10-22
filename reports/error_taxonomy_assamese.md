# Root-Cause Error Taxonomy Census: Assamese

- **Language**: Assamese
- **Model Evaluated**: `baseline_v1/checkpoints/final_assamese.pt` (Frozen Phase 0 Baseline)
- **Gold Diagnostic Population**: 270 sentences, 6732 tokens, 416 gold entities
- **Total Identified Errors**: 86

---

## 1. Ranked Error Taxonomy Table

| Rank | Error Category | Raw Count | % of Total Errors | % of Gold Entities | Statistical Status |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | **Complete Sentence Collapse** (`complete_sentence_collapse`) | **56 / 86** | 65.12% | 13.46% | **Robust (N >= 30)** |
| 2 | **Missed Entity** (`missed_entity`) | **14 / 86** | 16.28% | 3.37% | *Moderate (10 <= N < 30)* |
| 3 | **False Positive** (`false_positive`) | **10 / 86** | 11.63% | 2.40% | *Moderate (10 <= N < 30)* |
| 4 | **Incomplete Span** (`incomplete_span`) | **4 / 86** | 4.65% | 0.96% | **[Statistically Fragile (N < 10)]** |
| 5 | **Overextended Span** (`overextended_span`) | **2 / 86** | 2.33% | 0.48% | **[Statistically Fragile (N < 10)]** |
| 6 | **Wrong Entity Type** (`wrong_entity_type`) | **0 / 86** | 0.00% | 0.00% | **[Statistically Fragile (N < 10)]** |
| 7 | **Bi Transition Error** (`bi_transition_error`) | **0 / 86** | 0.00% | 0.00% | **[Statistically Fragile (N < 10)]** |
| 8 | **O Inside Entity** (`o_inside_entity`) | **0 / 86** | 0.00% | 0.00% | **[Statistically Fragile (N < 10)]** |

---

## 2. Held-Out Raw Corpus Diagnostic Census (Candidate Anomaly Quantification)

Predictions on unannotated raw sentences from the corpus pool (`data/processed/sentences.txt`),
screened for zero overlap against `test.conll` and used to quantify collapse and structural tag consistency in the wild:

- **Raw Sample Size**: 1500 sentences (22740 tokens)
- **Zero-Entity Predicted Sentences (Pure O)**: 1402 / 1500 (93.47%)
- **Sentences with Known Gazetteer Entities**: 262
- **Gazetteer Sentences with Complete Model Collapse**: 171 / 262 (65.27%)
- **Structural B-/I- Transition Errors in the Wild**: 0 (Zero violations; CRF Viterbi decoding maintains strict state machine validity)

---

## 3. Key Findings & Diagnostic Synthesis

1. **Dominant Failure Mode**: `complete_sentence_collapse` accounts for **56 / 86 (65.12%)** of all observed errors.
2. **Sentence Collapse Severity**: Complete-sentence collapse represents a severe vulnerability where the model outputs all-O despite valid entities being present.
3. **Absence of Transition Defects**: `bi_transition_error` is at 0 raw count across both gold and raw corpora, confirming the CRF transition parameterization is mathematically sound.
4. **Intervention Guidance for Phase 5**: Reliability engineering must prioritize sentence collapse recovery and lexical/inflectional false negative suppression.
