# Experiment 11B Report: Kokborok Human-In-The-Loop NER Annotation

**Generated**: 2026-09-30 14:57:48 UTC  
**Annotation Interface**: Local Web-Based HITL Annotation Tool  
**Gold-Test Protection**: SEALED & UNTOUCHED (Zero access during annotation)  

---

## 1. Important Scientific Disclosure

> [!IMPORTANT]
> **Scientific Integrity Statement**:
> The Experiment 11 annotations were machine-generated pre-annotations (produced via lexical dictionaries, suffix matching, and heuristics) and were **not treated as gold**. Experiment 11B introduces **genuine human verification**, wherein a human annotator makes the authoritative entity decision for every accepted sentence. Machine suggestions are presented strictly as unverified visual aids.

---

## 2. Annotation Progress & Summary Statistics

| Metric | Count | Percentage / Note |
|---|---|---|
| **Total Candidates in Master Pool** | **1367** | Sourced from Exp 10 deduplicated public corpus |
| **Total Reviewed by Human** | **0** | 0.0% review progress |
| **Human Verified (`HUMAN_VERIFIED`)** | **0** | Authoritative gold data exported to `kokborok_exp11b_human_verified.conll` |
| **Needs Review / Uncertain (`NEEDS_REVIEW`)** | **0** | Flagged for expert review; excluded from gold training pool |
| **Rejected Machine Suggestions (`REJECTED`)** | **0** | False positive machine triggers corrected or discarded |
| **Remaining Candidates (`UNREVIEWED`)** | **1367** | Pending human review queue |

---

## 3. Human-Verified Entity Distribution

*Applies exclusively to sentences with status `HUMAN_VERIFIED`:*

| Entity Type | Count | Share (%) |
|---|---|---|
| **PER** | 0 | 0.0% |
| **LOC** | 0 | 0.0% |
| **ORG** | 0 | 0.0% |
| **MISC** | 0 | 0.0% |
| **Total Entities** | **0** | 100.0% |
| **Multi-Token Entities** | **0** | 0.0% of entities |

---

## 4. Machine vs. Human Agreement & Decision Audit

| Human Decision | Count | Description |
|---|---|---|
| **ACCEPT** | 0 | Human validated machine suggestion without changes |
| **EDIT** | 0 | Human adjusted boundaries or modified entity type |
| **REJECT** | 0 | Human rejected machine suggestion (false positive) |
| **UNCERTAIN** | 0 | Human flagged sentence as ambiguous / needs native speaker review |

---

## 5. Artifacts and Auditability

- **Human Verified CoNLL**: [`data/processed/kokborok_exp11b_human_verified.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_exp11b_human_verified.conll)
- **Provenance CSV**: [`data/processed/kokborok_exp11b_provenance.csv`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_exp11b_provenance.csv)
- **Continuous Audit Log (JSONL)**: [`data/processed/kokborok_exp11b_annotation_log.jsonl`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_exp11b_annotation_log.jsonl)
- **Statistics JSON**: [`reports/kokborok_exp11b_annotation_statistics.json`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/kokborok_exp11b_annotation_statistics.json)
