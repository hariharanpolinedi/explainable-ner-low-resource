"""
Phase 15: Production Inference REST API (FastAPI).

Exposes lightweight, high-performance REST endpoints for multilingual NER inference
and attention attribution:
- POST /predict: Run NER tag extraction, span extraction, and attribution
- GET  /health: Health check, device reporting, and model cache status
- GET  /: API documentation and service metadata

Includes strict schema validation, clear error messages for unsupported languages,
and in-memory model caching with optional startup pre-warming.
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding for non-ASCII scripts on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from deployment.inference import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_BUNDLE_VERSION,
    SUPPORTED_LANGUAGES,
    get_predictor,
    select_device,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ner_api")


class PredictRequest(BaseModel):
    sentence: str = Field(
        ...,
        description="Raw input sentence in Konkani (Devanagari), Assamese, Sinhala, or Kokborok",
        example="ताचो जल्म नावेली साश्ट गोंय हांगा जालो .",
    )
    language: str = Field(
        ...,
        description="Target language code or name ('konkani', 'assamese', 'sinhala', 'kokborok')",
        example="konkani",
    )
    include_attribution: bool = Field(
        default=True,
        description="Whether to compute Gradient × Attention attribution for predicted entities",
    )
    bundle_version: str = Field(
        default=DEFAULT_BUNDLE_VERSION,
        description="Deployment bundle version (default: 'v1')",
    )


class EntitySpan(BaseModel):
    text: str
    type: str
    start_token: int
    end_token: int
    start_char: int
    end_char: int
    attributions: list[float] | None = None
    top_attended_tokens: list[dict[str, Any]] | None = None


class PredictResponse(BaseModel):
    status: str = "success"
    language: str
    device: str
    raw_sentence: str
    normalized_sentence: str
    tokens: list[str]
    tags: list[str]
    entities: list[EntitySpan]


class HealthResponse(BaseModel):
    status: str = "healthy"
    supported_languages: list[str]
    active_device: str
    device_name: str
    bundle_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional startup pre-warm
    logger.info("FastAPI service initializing...")
    yield
    logger.info("FastAPI service shutting down...")


app = FastAPI(
    title="Explainable Low-Resource NER Inference API",
    description=(
        "Production REST API for Explainable Named Entity Recognition on Konkani, "
        "Assamese, and Sinhala using Attention Attribution."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root() -> dict[str, Any]:
    """Service metadata and routing info."""
    return {
        "service": "Explainable Low-Resource NER Inference API",
        "phase": "Phase 15: Deployment & Packaging",
        "languages": list(SUPPORTED_LANGUAGES),
        "endpoints": {
            "/predict": "POST - Run NER inference and attribution",
            "/health": "GET - System health and device status",
            "/docs": "GET - Interactive Swagger UI",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Check health, device availability, and supported languages."""
    import torch
    dev = select_device()
    dev_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    return HealthResponse(
        status="healthy",
        supported_languages=list(SUPPORTED_LANGUAGES),
        active_device=str(dev),
        device_name=dev_name,
        bundle_version=DEFAULT_BUNDLE_VERSION,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_endpoint(req: PredictRequest) -> PredictResponse:
    """
    Run named entity recognition and attention attribution on input text.
    """
    lang_clean = req.language.lower().strip()
    if lang_clean not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported language '{req.language}'. "
                f"Must be one of {list(SUPPORTED_LANGUAGES)}."
            ),
        )

    try:
        predictor = get_predictor(
            language=lang_clean,
            bundle_version=req.bundle_version,
        )
        res = predictor.predict(
            sentence=req.sentence,
            include_attribution=req.include_attribution,
        )
        return PredictResponse(
            status="success",
            language=res["language"],
            device=res["device"],
            raw_sentence=res["raw_sentence"],
            normalized_sentence=res["normalized_sentence"],
            tokens=res["tokens"],
            tags=res["tags"],
            entities=res["entities"],
        )
    except Exception as exc:
        logger.error("Inference failure for language %s: %s", lang_clean, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution error: {exc}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the NER Inference REST API")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--prewarm", action="store_true", help="Pre-load all models into cache at startup")
    args = parser.parse_args()

    if args.prewarm:
        print("Pre-warming model cache across all languages...")
        for l in SUPPORTED_LANGUAGES:
            get_predictor(l)
        print("All models pre-warmed.")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
