"""
Database and storage layer for Kokborok Human-In-The-Loop Annotation Tool.
Maintains continuous SQLite state and JSONL append-only audit logging.
"""

import json
import sqlite3
import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from annotation_tool.validator import validate_bio_sequence, bio_to_spans, spans_to_bio

PROJECT_ROOT = Path("c:/Users/harih/Desktop/NLP-MARK-3")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
DB_PATH = DATA_DIR / "kokborok_exp11b_annotations.db"
LOG_JSONL_PATH = DATA_DIR / "kokborok_exp11b_annotation_log.jsonl"


def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH):
    """Initializes the SQLite schema."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sentence_id TEXT UNIQUE NOT NULL,
        source TEXT,
        source_file TEXT,
        source_row_id TEXT,
        license TEXT,
        raw_text TEXT NOT NULL,
        tokens_json TEXT NOT NULL,
        machine_tags_json TEXT NOT NULL,
        machine_confidence TEXT DEFAULT 'MACHINE_PREANNOTATION',
        human_tags_json TEXT,
        human_status TEXT DEFAULT 'UNREVIEWED', -- UNREVIEWED, HUMAN_VERIFIED, NEEDS_REVIEW, REJECTED
        decision TEXT, -- ACCEPT, EDIT, REJECT, UNCERTAIN
        annotator_id TEXT,
        notes TEXT,
        last_updated TEXT,
        is_locked_test INTEGER DEFAULT 0 -- Safety flag
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sentence_id TEXT NOT NULL,
        annotator_id TEXT NOT NULL,
        action TEXT NOT NULL,
        decision TEXT,
        status TEXT,
        machine_tags_json TEXT NOT NULL,
        human_tags_json TEXT,
        notes TEXT,
        timestamp TEXT NOT NULL
    );
    """)

    conn.commit()
    conn.close()


def load_raw_candidates(db_path: Path = DB_PATH, force_reload: bool = False):
    """
    Ingests the candidate sentences from kokborok_exp11_provenance.csv & pre-annotated CoNLLs
    into the database if not already populated.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM candidates")
    count = cur.fetchone()[0]
    if count > 0 and not force_reload:
        conn.close()
        return count

    # Read pre-annotated CoNLL blocks to map sentence_id / text -> machine tags
    preannotated_files = [
        DATA_DIR / "kokborok_exp11_annotated.conll",
        DATA_DIR / "kokborok_exp11_review_queue.conll",
        DATA_DIR / "kokborok_exp11_uncertain.conll"
    ]

    machine_tags_map = {}
    for pfile in preannotated_files:
        if not pfile.exists():
            continue
        with open(pfile, "r", encoding="utf-8") as f:
            content = f.read().strip()
        blocks = content.split("\n\n")
        for blk in blocks:
            lines = blk.strip().split("\n")
            if not lines or lines[0].startswith("# KOKBOROK"):
                continue
            meta_line = lines[0] if lines[0].startswith("# id:") else ""
            token_lines = lines[1:] if meta_line else lines
            sent_id = ""
            if meta_line:
                m_id = re.search(r'# id:\s*([^\s|]+)', meta_line)
                if m_id:
                    sent_id = m_id.group(1).strip()
            
            toks = []
            tags = []
            for tl in token_lines:
                parts = tl.strip().split()
                if len(parts) >= 2:
                    toks.append(parts[0])
                    tags.append(parts[1])
                elif len(parts) == 1:
                    toks.append(parts[0])
                    tags.append("O")
            
            if toks and sent_id:
                machine_tags_map[sent_id] = (toks, tags)

    # Read candidate master file or provenance
    cand_file = DATA_DIR / "kokborok_ner_candidate_sentences.conll"
    with open(cand_file, "r", encoding="utf-8") as f:
        cand_blocks = f.read().strip().split("\n\n")

    # Load provenance CSV for metadata
    prov_file = DATA_DIR / "kokborok_exp11_provenance.csv"
    prov_meta = {}
    if prov_file.exists():
        with open(prov_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                prov_meta[r.get("sentence_id", "")] = r

    inserted = 0
    for block in cand_blocks:
        lines = block.strip().split("\n")
        if not lines or lines[0].startswith("# KOKBOROK"):
            continue
        meta_line = lines[0] if lines[0].startswith("# id:") else ""
        token_lines = lines[1:] if meta_line else lines

        sent_id = ""
        source = ""
        if meta_line:
            m_id = re.search(r'# id:\s*([^\s|]+)', meta_line)
            m_src = re.search(r'source:\s*([^|]+)', meta_line)
            if m_id: sent_id = m_id.group(1).strip()
            if m_src: source = m_src.group(1).strip()

        toks = []
        for tl in token_lines:
            parts = tl.strip().split()
            if parts:
                toks.append(parts[0])

        if not toks:
            continue

        raw_text = " ".join(toks)
        if not sent_id:
            sent_id = f"KOK_CAND_{inserted+1:05d}"

        # Check if we have pre-annotated machine tags
        if sent_id in machine_tags_map:
            _, m_tags = machine_tags_map[sent_id]
            if len(m_tags) != len(toks):
                m_tags = ["O"] * len(toks)
        else:
            m_tags = ["O"] * len(toks)

        p_info = prov_meta.get(sent_id, {})
        source_file = p_info.get("source_file", source)
        source_row_id = p_info.get("source_row_id", "")
        license_str = p_info.get("license", "Public / Academic Open Access")

        cur.execute("""
        INSERT OR REPLACE INTO candidates (
            sentence_id, source, source_file, source_row_id, license,
            raw_text, tokens_json, machine_tags_json, machine_confidence,
            human_tags_json, human_status, decision, annotator_id, notes,
            last_updated, is_locked_test
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sent_id, source, source_file, source_row_id, license_str,
            raw_text, json.dumps(toks, ensure_ascii=False),
            json.dumps(m_tags, ensure_ascii=False),
            "MACHINE_PREANNOTATION",
            None,
            "UNREVIEWED",
            None,
            None,
            "",
            None,
            0
        ))
        inserted += 1

    conn.commit()
    conn.close()
    return inserted


def get_candidate(sentence_id: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM candidates WHERE sentence_id = ?", (sentence_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["tokens"] = json.loads(d["tokens_json"])
    d["machine_tags"] = json.loads(d["machine_tags_json"])
    d["human_tags"] = json.loads(d["human_tags_json"]) if d["human_tags_json"] else None
    
    # Also provide spans for UI convenience
    m_spans, _ = bio_to_spans(d["tokens"], d["machine_tags"])
    d["machine_spans"] = m_spans
    if d["human_tags"]:
        h_spans, _ = bio_to_spans(d["tokens"], d["human_tags"])
        d["human_spans"] = h_spans
    else:
        d["human_spans"] = None
    return d


def get_candidate_by_index(index: int, filter_status: Optional[str] = None, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    if filter_status and filter_status != "ALL":
        cur.execute("SELECT sentence_id FROM candidates WHERE human_status = ? ORDER BY id LIMIT 1 OFFSET ?", (filter_status, index))
    else:
        cur.execute("SELECT sentence_id FROM candidates ORDER BY id LIMIT 1 OFFSET ?", (index,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return get_candidate(row["sentence_id"], db_path)


def get_stats(db_path: Path = DB_PATH) -> Dict[str, Any]:
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM candidates")
    total_candidates = cur.fetchone()[0]

    cur.execute("SELECT human_status, COUNT(*) FROM candidates GROUP BY human_status")
    status_counts = dict(cur.fetchall())

    reviewed = status_counts.get("HUMAN_VERIFIED", 0) + status_counts.get("NEEDS_REVIEW", 0) + status_counts.get("REJECTED", 0)
    human_verified = status_counts.get("HUMAN_VERIFIED", 0)
    uncertain = status_counts.get("NEEDS_REVIEW", 0)
    rejected = status_counts.get("REJECTED", 0)
    unreviewed = status_counts.get("UNREVIEWED", 0)

    # Count verified entities by type
    cur.execute("SELECT tokens_json, human_tags_json FROM candidates WHERE human_status = 'HUMAN_VERIFIED'")
    rows = cur.fetchall()
    
    ent_counts = {"PER": 0, "LOC": 0, "ORG": 0, "MISC": 0}
    multi_token_count = 0
    total_entities = 0

    for r in rows:
        toks = json.loads(r["tokens_json"])
        htags = json.loads(r["human_tags_json"]) if r["human_tags_json"] else []
        spans, _ = bio_to_spans(toks, htags)
        for s in spans:
            stype = s["type"]
            if stype in ent_counts:
                ent_counts[stype] += 1
            total_entities += 1
            if (s["end"] - s["start"]) > 1:
                multi_token_count += 1

    # Machine/Human Agreement analysis
    cur.execute("""
    SELECT machine_tags_json, human_tags_json, decision 
    FROM candidates 
    WHERE human_status IN ('HUMAN_VERIFIED', 'REJECTED', 'NEEDS_REVIEW')
    """)
    dec_rows = cur.fetchall()
    agreements = 0
    disagreements = 0
    decisions_summary = {}

    for dr in dec_rows:
        mt = dr["machine_tags_json"]
        ht = dr["human_tags_json"]
        dec = dr["decision"]
        decisions_summary[dec] = decisions_summary.get(dec, 0) + 1
        if mt == ht:
            agreements += 1
        else:
            disagreements += 1

    conn.close()

    return {
        "total_candidates": total_candidates,
        "reviewed": reviewed,
        "human_verified": human_verified,
        "uncertain": uncertain,
        "rejected": rejected,
        "unreviewed": unreviewed,
        "entities": {
            "PER": ent_counts["PER"],
            "LOC": ent_counts["LOC"],
            "ORG": ent_counts["ORG"],
            "MISC": ent_counts["MISC"],
            "total": total_entities,
            "multi_token": multi_token_count
        },
        "agreement": {
            "matches": agreements,
            "disagreements": disagreements,
            "decisions": decisions_summary
        }
    }


def save_human_annotation(
    sentence_id: str,
    annotator_id: str,
    decision: str,  # ACCEPT, EDIT, REJECT, UNCERTAIN
    human_tags: Optional[List[str]] = None,
    notes: str = "",
    db_path: Path = DB_PATH
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Saves a human annotation with strict status determination.
    """
    if not annotator_id or not annotator_id.strip():
        return False, "Annotator ID cannot be empty.", {}

    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT * FROM candidates WHERE sentence_id = ?", (sentence_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False, f"Sentence ID {sentence_id} not found.", {}

    toks = json.loads(row["tokens_json"])
    machine_tags = json.loads(row["machine_tags_json"])

    # Determine tags and status based on decision
    now_ts = datetime.now(timezone.utc).isoformat()

    if decision == "ACCEPT":
        final_tags = machine_tags if human_tags is None else human_tags
        status = "HUMAN_VERIFIED"
    elif decision == "EDIT":
        if human_tags is None:
            conn.close()
            return False, "EDIT decision requires human_tags.", {}
        final_tags = human_tags
        status = "HUMAN_VERIFIED"
    elif decision == "REJECT":
        # Human rejected machine suggestion; tags default to all O unless explicitly edited to O
        final_tags = ["O"] * len(toks) if human_tags is None else human_tags
        status = "REJECTED"
    elif decision == "UNCERTAIN":
        final_tags = machine_tags if human_tags is None else human_tags
        status = "NEEDS_REVIEW"
    else:
        conn.close()
        return False, f"Invalid decision '{decision}'. Must be ACCEPT, EDIT, REJECT, or UNCERTAIN.", {}

    # Validate BIO tags
    is_valid, errors = validate_bio_sequence(toks, final_tags)
    if not is_valid:
        conn.close()
        return False, f"BIO Validation Failed: {'; '.join(errors)}", {}

    # Save to SQLite
    cur.execute("""
    UPDATE candidates SET
        human_tags_json = ?,
        human_status = ?,
        decision = ?,
        annotator_id = ?,
        notes = ?,
        last_updated = ?
    WHERE sentence_id = ?
    """, (
        json.dumps(final_tags, ensure_ascii=False),
        status,
        decision,
        annotator_id.strip(),
        notes,
        now_ts,
        sentence_id
    ))

    # Add to audit log table
    cur.execute("""
    INSERT INTO audit_logs (
        sentence_id, annotator_id, action, decision, status,
        machine_tags_json, human_tags_json, notes, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        sentence_id,
        annotator_id.strip(),
        "SAVE_ANNOTATION",
        decision,
        status,
        json.dumps(machine_tags, ensure_ascii=False),
        json.dumps(final_tags, ensure_ascii=False),
        notes,
        now_ts
    ))

    conn.commit()
    conn.close()

    # Append to JSONL audit log
    log_entry = {
        "sentence_id": sentence_id,
        "annotator_id": annotator_id.strip(),
        "timestamp": now_ts,
        "decision": decision,
        "status": status,
        "tokens": toks,
        "machine_tags": machine_tags,
        "human_tags": final_tags,
        "notes": notes
    }
    with open(LOG_JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    updated_candidate = get_candidate(sentence_id, db_path)
    return True, "Annotation successfully saved.", updated_candidate
