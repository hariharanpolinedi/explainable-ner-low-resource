# Inter-Annotator Agreement Report (Phase 3)

**Date**: 2026-09-16  
**Scope**: 100-sentence overlap dual-annotation sample for Konkani, Assamese, Sinhala  
**Acceptance Threshold**: Cohen's κ >= 0.75  

---

## 1. Summary of Agreement Metrics

| Language | Overlap Sents | Total Tokens | Observed Agreement (Po) | Chance Agreement (Pe) | Cohen's Kappa (κ) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Konkani** | 100 | 1,794 | 99.89% | 85.51% | **0.9923** | **PASS (>=0.75)** |
| **Assamese** | 100 | 1,788 | 99.66% | 87.18% | **0.9738** | **PASS (>=0.75)** |
| **Sinhala** | 100 | 2,113 | 98.86% | 83.38% | **0.9317** | **PASS (>=0.75)** |

---

## 2. Disagreement Analysis & Adjudication Protocol

- **Primary Disagreement Source**: Boundary assignment on honorifics and occupational titles preceding person names (e.g., Annotator B occasionally tagged title prefix as part of the `B-PER` span, whereas Annotator A excluded it as `O`).
- **Adjudication Standard**: Disagreements were systematically resolved by enforcing the written guideline rules: standalone titles and honorifics are strictly categorized as `O`, reserving `B-PER` exclusively for the personal proper name.
- **Guideline Refinement**: The disambiguation rule in `annotation/guidelines_{lang}.md` was codified and verified against all disputed spans.

---

## 3. Sample Adjudicated Disagreements

### Konkani
| Sent # | Token | Annotator A | Annotator B | Adjudicated | Resolution Rule |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 92 | `गोंय` | `B-ORG` | `B-LOC` | `B-ORG` | Institutional entity prioritized as ORG over nested LOC (Rule ORG-2) |
| 92 | `विद्यापीठ` | `I-ORG` | `B-ORG` | `I-ORG` | Codified disambiguation guideline rule applied |

### Assamese
| Sent # | Token | Annotator A | Annotator B | Adjudicated | Resolution Rule |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 23 | `অসম` | `B-ORG` | `B-LOC` | `B-ORG` | Institutional entity prioritized as ORG over nested LOC (Rule ORG-2) |
| 23 | `চৰকাৰ` | `I-ORG` | `B-ORG` | `I-ORG` | Codified disambiguation guideline rule applied |
| 34 | `অসম` | `B-ORG` | `B-LOC` | `B-ORG` | Institutional entity prioritized as ORG over nested LOC (Rule ORG-2) |
| 34 | `চৰকাৰ` | `I-ORG` | `B-ORG` | `I-ORG` | Codified disambiguation guideline rule applied |
| 96 | `অসম` | `B-ORG` | `B-LOC` | `B-ORG` | Institutional entity prioritized as ORG over nested LOC (Rule ORG-2) |

### Sinhala
| Sent # | Token | Annotator A | Annotator B | Adjudicated | Resolution Rule |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0 | `ජනාධිපති` | `O` | `B-PER` | `O` | Honorific/title excluded from proper name span (Rule PER-1) |
| 0 | `මහින්ද` | `B-PER` | `I-PER` | `B-PER` | Honorific/title excluded from proper name span (Rule PER-1) |
| 12 | `ශ්‍රී` | `B-ORG` | `B-LOC` | `B-ORG` | Institutional entity prioritized as ORG over nested LOC (Rule ORG-2) |
| 12 | `ලංකා` | `I-ORG` | `B-ORG` | `I-ORG` | Codified disambiguation guideline rule applied |
| 25 | `ශ්‍රී` | `B-ORG` | `B-LOC` | `B-ORG` | Institutional entity prioritized as ORG over nested LOC (Rule ORG-2) |

