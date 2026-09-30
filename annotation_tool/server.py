"""
FastAPI Server for Kokborok Human-In-The-Loop Annotation Tool (Exp 11B).
Provides REST endpoints and serves static UI files.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

PROJECT_ROOT = Path("c:/Users/harih/Desktop/NLP-MARK-3")
sys.path.insert(0, str(PROJECT_ROOT))

from annotation_tool.db import (
    init_db,
    load_raw_candidates,
    get_candidate,
    get_candidate_by_index,
    get_stats,
    save_human_annotation,
    DB_PATH
)
from annotation_tool.validator import spans_to_bio, bio_to_spans, validate_bio_sequence
from annotation_tool.exporter import export_verified_datasets

app = FastAPI(title="Kokborok HITL NER Annotation Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = PROJECT_ROOT / "annotation_tool" / "static"


class SaveAnnotationRequest(BaseModel):
    sentence_id: str
    annotator_id: str
    decision: str  # ACCEPT, EDIT, REJECT, UNCERTAIN
    spans: Optional[List[Dict[str, Any]]] = None  # [{"start": 0, "end": 2, "type": "ORG"}]
    human_tags: Optional[List[str]] = None
    notes: Optional[str] = ""


@app.on_event("startup")
def startup_event():
    init_db(DB_PATH)
    load_raw_candidates(DB_PATH)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "project": "Kokborok NER HITL", "version": "1.0.0"}


@app.get("/api/stats")
def fetch_stats():
    return get_stats(DB_PATH)


@app.get("/api/candidate")
def fetch_candidate(
    sentence_id: Optional[str] = None,
    index: Optional[int] = None,
    filter_status: Optional[str] = "ALL"
):
    if sentence_id:
        cand = get_candidate(sentence_id, DB_PATH)
    elif index is not None:
        cand = get_candidate_by_index(index, filter_status, DB_PATH)
    else:
        cand = get_candidate_by_index(0, filter_status, DB_PATH)

    if not cand:
        raise HTTPException(status_code=404, detail="Candidate sentence not found.")
    return cand


@app.post("/api/save")
def submit_annotation(req: SaveAnnotationRequest):
    # Fetch candidate to get tokens
    cand = get_candidate(req.sentence_id, DB_PATH)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate sentence not found.")

    tokens = cand["tokens"]

    # If spans were passed instead of human_tags, convert spans to BIO
    if req.spans is not None:
        tags, errors = spans_to_bio(tokens, req.spans)
        if errors:
            raise HTTPException(status_code=400, detail=f"Span conversion error: {'; '.join(errors)}")
        final_tags = tags
    elif req.human_tags is not None:
        final_tags = req.human_tags
    else:
        final_tags = None

    success, msg, updated_cand = save_human_annotation(
        sentence_id=req.sentence_id,
        annotator_id=req.annotator_id,
        decision=req.decision,
        human_tags=final_tags,
        notes=req.notes or "",
        db_path=DB_PATH
    )

    if not success:
        raise HTTPException(status_code=400, detail=msg)

    # Automatically trigger export refresh
    stats = export_verified_datasets(DB_PATH)

    return {
        "success": True,
        "message": msg,
        "candidate": updated_cand,
        "stats": stats
    }


@app.post("/api/export")
def trigger_export():
    stats = export_verified_datasets(DB_PATH)
    return {
        "success": True,
        "message": "Datasets and reports exported successfully.",
        "stats": stats
    }


# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "Kokborok HITL UI Static assets loading..."})


def run_server(port: int = 8000, host: str = "127.0.0.1"):
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
