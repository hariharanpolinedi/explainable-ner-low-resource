# Root-Cause Error Taxonomy Census: Sinhala

- **Language**: Sinhala
- **Model Evaluated**: `baseline_v1/checkpoints/final_sinhala.pt` (Frozen Phase 0 Baseline)
- **Gold Diagnostic Population**: 263 sentences, 6762 tokens, 300 gold entities
- **Total Identified Errors**: 55

---

## 1. Ranked Error Taxonomy Table

| Rank | Error Category | Raw Count | % of Total Errors | % of Gold Entities | Statistical Status |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | **Complete Sentence Collapse** (`complete_sentence_collapse`) | **28 / 55** | 50.91% | 9.33% | *Moderate (10 <= N < 30)* |
| 2 | **Missed Entity** (`missed_entity`) | **15 / 55** | 27.27% | 5.00% | *Moderate (10 <= N < 30)* |
| 3 | **False Positive** (`false_positive`) | **12 / 55** | 21.82% | 4.00% | *Moderate (10 <= N < 30)* |
| 4 | **Wrong Entity Type** (`wrong_entity_type`) | **0 / 55** | 0.00% | 0.00% | **[Statistically Fragile (N < 10)]** |
| 5 | **Incomplete Span** (`incomplete_span`) | **0 / 55** | 0.00% | 0.00% | **[Statistically Fragile (N < 10)]** |
| 6 | **Overextended Span** (`overextended_span`) | **0 / 55** | 0.00% | 0.00% | **[Statistically Fragile (N < 10)]** |
| 7 | **Bi Transition Error** (`bi_transition_error`) | **0 / 55** | 0.00% | 0.00% | **[Statistically Fragile (N < 10)]** |
| 8 | **O Inside Entity** (`o_inside_entity`) | **0 / 55** | 0.00% | 0.00% | **[Statistically Fragile (N < 10)]** |

---

## 2. Held-Out Raw Corpus Diagnostic Census (Candidate Anomaly Quantification)

Predictions on unannotated raw sentences from the corpus pool (`data/processed/sentences.txt`),
screened for zero overlap against `test.conll` and used to quantify collapse and structural tag consistency in the wild:

- **Raw Sample Size**: 1500 sentences (22686 tokens)
- **Zero-Entity Predicted Sentences (Pure O)**: 1457 / 1500 (97.13%)
- **Sentences with Known Gazetteer Entities**: 117
- **Gazetteer Sentences with Complete Model Collapse**: 79 / 117 (67.52%)
- **Structural B-/I- Transition Errors in the Wild**: 0 (Zero violations; CRF Viterbi decoding maintains strict state machine validity)

---

## 3. Key Findings & Diagnostic Synthesis

1. **Dominant Failure Mode**: `complete_sentence_collapse` accounts for **28 / 55 (50.91%)** of all observed errors.
2. **Sentence Collapse Severity**: Complete-sentence collapse represents a severe vulnerability where the model outputs all-O despite valid entities being present.
3. **Absence of Transition Defects**: `bi_transition_error` is at 0 raw count across both gold and raw corpora, confirming the CRF transition parameterization is mathematically sound.
4. **Intervention Guidance for Phase 5**: Reliability engineering must prioritize sentence collapse recovery and lexical/inflectional false negative suppression.
