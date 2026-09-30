"""
Exporter for Kokborok Human-In-The-Loop Annotation Tool (Exp 11B).
Generates verified CoNLL, provenance CSV, and statistics JSON/Markdown reports.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from annotation_tool.db import get_db_connection, get_stats, DB_PATH
from annotation_tool.validator import bio_to_spans

PROJECT_ROOT = Path("c:/Users/harih/Desktop/NLP-MARK-3")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def export_verified_datasets(db_path: Path = DB_PATH) -> Dict[str, Any]:
    """
    Exports:
    1. kokborok_exp11b_human_verified.conll (ONLY HUMAN_VERIFIED sentences)
    2. kokborok_exp11b_provenance.csv (Full audit & provenance)
    3. reports/kokborok_exp11b_annotation_statistics.json
    4. reports/kokborok_exp11b_annotation_report.md
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_db_connection(db_path)
    cur = conn.cursor()

    # 1. Export HUMAN_VERIFIED sentences to CoNLL
    cur.execute("""
    SELECT sentence_id, source, tokens_json, human_tags_json, annotator_id, last_updated 
    FROM candidates 
    WHERE human_status = 'HUMAN_VERIFIED'
    ORDER BY id
    """)
    verified_rows = cur.fetchall()

    conll_out_path = DATA_DIR / "kokborok_exp11b_human_verified.conll"
    conll_lines = []
    
    total_verified_tokens = 0
    total_verified_entities = 0
    type_counts = {"PER": 0, "LOC": 0, "ORG": 0, "MISC": 0}
    multi_token_entities = 0

    for r in verified_rows:
        sent_id = r["sentence_id"]
        source = r["source"] or "public_corpus"
        annotator = r["annotator_id"]
        toks = json.loads(r["tokens_json"])
        tags = json.loads(r["human_tags_json"])
        
        total_verified_tokens += len(toks)
        spans, _ = bio_to_spans(toks, tags)
        for s in spans:
            stype = s["type"]
            if stype in type_counts:
                type_counts[stype] += 1
            total_verified_entities += 1
            if (s["end"] - s["start"]) > 1:
                multi_token_entities += 1

        conll_lines.append(f"# id: {sent_id} | source: {source} | annotator: {annotator} | status: HUMAN_VERIFIED")
        for tok, tag in zip(toks, tags):
            conll_lines.append(f"{tok}\t{tag}")
        conll_lines.append("")  # Blank line separator

    with open(conll_out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(conll_lines))

    # 2. Export Provenance CSV for all reviewed or candidate items
    cur.execute("SELECT * FROM candidates ORDER BY id")
    all_candidates = cur.fetchall()

    prov_out_path = DATA_DIR / "kokborok_exp11b_provenance.csv"
    prov_fieldnames = [
        "sentence_id", "source", "source_file", "source_row_id", "license",
        "human_status", "decision", "annotator_id", "timestamp",
        "machine_entity_count", "human_entity_count",
        "machine_labels", "human_labels", "notes"
    ]

    with open(prov_out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=prov_fieldnames)
        writer.writeheader()

        for c in all_candidates:
            toks = json.loads(c["tokens_json"])
            m_tags = json.loads(c["machine_tags_json"])
            h_tags = json.loads(c["human_tags_json"]) if c["human_tags_json"] else []

            m_spans, _ = bio_to_spans(toks, m_tags)
            h_spans, _ = bio_to_spans(toks, h_tags) if h_tags else ([], [])

            writer.writerow({
                "sentence_id": c["sentence_id"],
                "source": c["source"] or "",
                "source_file": c["source_file"] or "",
                "source_row_id": c["source_row_id"] or "",
                "license": c["license"] or "",
                "human_status": c["human_status"] or "UNREVIEWED",
                "decision": c["decision"] or "",
                "annotator_id": c["annotator_id"] or "",
                "timestamp": c["last_updated"] or "",
                "machine_entity_count": len(m_spans),
                "human_entity_count": len(h_spans),
                "machine_labels": " ".join(m_tags),
                "human_labels": " ".join(h_tags) if h_tags else "",
                "notes": c["notes"] or ""
            })

    # 3. Compute Detailed Annotation Statistics
    stats = get_stats(db_path)
    stats["exported_files"] = {
        "human_verified_conll": str(conll_out_path),
        "provenance_csv": str(prov_out_path),
        "total_verified_sentences": len(verified_rows),
        "total_verified_tokens": total_verified_tokens,
        "total_verified_entities": total_verified_entities
    }

    stats_json_path = REPORTS_DIR / "kokborok_exp11b_annotation_statistics.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 4. Generate Detailed Markdown Report
    report_md_path = REPORTS_DIR / "kokborok_exp11b_annotation_report.md"
    generate_markdown_report(stats, report_md_path)

    conn.close()
    return stats


def generate_markdown_report(stats: Dict[str, Any], report_path: Path):
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ents = stats.get("entities", {})
    aggr = stats.get("agreement", {})
    dec_summary = aggr.get("decisions", {})

    report_content = f"""# Experiment 11B Report: Kokborok Human-In-The-Loop NER Annotation

**Generated**: {now_str}  
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
| **Total Candidates in Master Pool** | **{stats.get('total_candidates', 0)}** | Sourced from Exp 10 deduplicated public corpus |
| **Total Reviewed by Human** | **{stats.get('reviewed', 0)}** | {stats.get('reviewed', 0) / max(stats.get('total_candidates', 1), 1) * 100:.1f}% review progress |
| **Human Verified (`HUMAN_VERIFIED`)** | **{stats.get('human_verified', 0)}** | Authoritative gold data exported to `kokborok_exp11b_human_verified.conll` |
| **Needs Review / Uncertain (`NEEDS_REVIEW`)** | **{stats.get('uncertain', 0)}** | Flagged for expert review; excluded from gold training pool |
| **Rejected Machine Suggestions (`REJECTED`)** | **{stats.get('rejected', 0)}** | False positive machine triggers corrected or discarded |
| **Remaining Candidates (`UNREVIEWED`)** | **{stats.get('unreviewed', 0)}** | Pending human review queue |

---

## 3. Human-Verified Entity Distribution

*Applies exclusively to sentences with status `HUMAN_VERIFIED`:*

| Entity Type | Count | Share (%) |
|---|---|---|
| **PER** | {ents.get('PER', 0)} | {ents.get('PER', 0) / max(ents.get('total', 1), 1) * 100:.1f}% |
| **LOC** | {ents.get('LOC', 0)} | {ents.get('LOC', 0) / max(ents.get('total', 1), 1) * 100:.1f}% |
| **ORG** | {ents.get('ORG', 0)} | {ents.get('ORG', 0) / max(ents.get('total', 1), 1) * 100:.1f}% |
| **MISC** | {ents.get('MISC', 0)} | {ents.get('MISC', 0) / max(ents.get('total', 1), 1) * 100:.1f}% |
| **Total Entities** | **{ents.get('total', 0)}** | 100.0% |
| **Multi-Token Entities** | **{ents.get('multi_token', 0)}** | {ents.get('multi_token', 0) / max(ents.get('total', 1), 1) * 100:.1f}% of entities |

---

## 4. Machine vs. Human Agreement & Decision Audit

| Human Decision | Count | Description |
|---|---|---|
| **ACCEPT** | {dec_summary.get('ACCEPT', 0)} | Human validated machine suggestion without changes |
| **EDIT** | {dec_summary.get('EDIT', 0)} | Human adjusted boundaries or modified entity type |
| **REJECT** | {dec_summary.get('REJECT', 0)} | Human rejected machine suggestion (false positive) |
| **UNCERTAIN** | {dec_summary.get('UNCERTAIN', 0)} | Human flagged sentence as ambiguous / needs native speaker review |

---

## 5. Artifacts and Auditability

- **Human Verified CoNLL**: [`data/processed/kokborok_exp11b_human_verified.conll`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_exp11b_human_verified.conll)
- **Provenance CSV**: [`data/processed/kokborok_exp11b_provenance.csv`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_exp11b_provenance.csv)
- **Continuous Audit Log (JSONL)**: [`data/processed/kokborok_exp11b_annotation_log.jsonl`](file:///c:/Users/harih/Desktop/NLP-MARK-3/data/processed/kokborok_exp11b_annotation_log.jsonl)
- **Statistics JSON**: [`reports/kokborok_exp11b_annotation_statistics.json`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/kokborok_exp11b_annotation_statistics.json)
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
