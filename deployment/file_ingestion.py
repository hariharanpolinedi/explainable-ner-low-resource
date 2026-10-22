"""
Multilingual Document & Image Ingestion Engine.
Extracts clean Unicode text from:
1. PDF Documents (.pdf) - Digital stream extraction + Scanned OCR fallback
2. Images / Photos (.png, .jpg, .jpeg, .webp, .bmp, .tiff) - OCR with Indic + Latin support
3. Word Documents (.docx) - Paragraph & table text extraction
4. Plain Text (.txt, .md, .csv, .json) - Multi-codec Unicode decoding

Also provides:
- render_pdf_with_highlights(): annotates PDF pages with entity highlight boxes, returns PNG bytes per page
- render_image_with_highlights(): annotates an OCR image with entity highlight boxes, returns PNG bytes
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

# Drawing library
try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
except ImportError:
    PILImage = None
    ImageDraw = None
    ImageFont = None

# PDF & Docx libraries
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Global EasyOCR reader cache (lazy loaded)
_OCR_READERS: dict[str, Any] = {}


def get_ocr_reader(lang_group: str = "indic") -> Any:
    """Lazy initialization of EasyOCR reader with GPU acceleration."""
    global _OCR_READERS
    if lang_group in _OCR_READERS:
        return _OCR_READERS[lang_group]

    try:
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()

        if lang_group == "indic":
            # Supports English, Devanagari (Konkani/Hindi), Bengali (Assamese/Bengali)
            reader = easyocr.Reader(["en", "hi", "bn"], gpu=use_gpu)
        else:
            reader = easyocr.Reader(["en"], gpu=use_gpu)

        _OCR_READERS[lang_group] = reader
        return reader
    except Exception as e:
        print(f"[WARN] Failed to initialize EasyOCR for {lang_group}: {e}")
        return None


def extract_from_pdf(file_bytes: bytes) -> dict[str, Any]:
    """Extract text from PDF pages using PyMuPDF with layout awareness."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed.")

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    total_text = []
    scanned_pages = []

    for i, page in enumerate(doc):
        page_text = page.get_text("text").strip()
        if not page_text:
            # Page might be a scanned image
            scanned_pages.append(i + 1)
        else:
            pages.append({"page_num": i + 1, "text": page_text})
            total_text.append(page_text)

    # If document has empty scanned pages and OCR is available, run OCR on those pages
    warnings = []
    if scanned_pages:
        warnings.append(f"Pages {scanned_pages} had no selectable digital text; attempting OCR...")
        reader = get_ocr_reader("indic")
        if reader and Image:
            for page_idx in scanned_pages:
                page = doc[page_idx - 1]
                pix = page.get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_results = reader.readtext(pix.tobytes("png"), detail=0)
                page_text = " ".join(ocr_results).strip()
                if page_text:
                    pages.append({"page_num": page_idx, "text": page_text})
                    total_text.append(page_text)

    # Sort pages by page_num
    pages = sorted(pages, key=lambda p: p["page_num"])
    joined_text = "\n\n".join(p["text"] for p in pages).strip()

    return {
        "text": joined_text,
        "pages": pages,
        "num_pages": len(doc),
        "file_type": "pdf",
        "warnings": warnings,
    }


def extract_from_image(file_bytes: bytes) -> dict[str, Any]:
    """Extract text from an image/photo using EasyOCR."""
    reader = get_ocr_reader("indic")
    if not reader:
        # Fallback to English reader
        reader = get_ocr_reader("en")

    if not reader:
        raise RuntimeError("OCR engine is not available. Please verify easyocr installation.")

    results = reader.readtext(file_bytes, detail=0)
    text = " ".join(results).strip()

    return {
        "text": text,
        "pages": [{"page_num": 1, "text": text}],
        "num_pages": 1,
        "file_type": "image",
        "warnings": [],
    }


def extract_from_docx(file_bytes: bytes) -> dict[str, Any]:
    """Extract paragraphs and table text from a Word document."""
    if docx is None:
        raise RuntimeError("python-docx is not installed.")

    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = []

    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            paragraphs.append(t)

    # Also extract table text
    for table in doc.tables:
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_cells:
                paragraphs.append(" | ".join(row_cells))

    joined_text = "\n\n".join(paragraphs).strip()
    return {
        "text": joined_text,
        "pages": [{"page_num": 1, "text": joined_text}],
        "num_pages": 1,
        "file_type": "docx",
        "warnings": [],
    }


def extract_from_text(file_bytes: bytes) -> dict[str, Any]:
    """Decode raw bytes using multi-codec fallback for Unicode, UTF-8, UTF-16, Latin-1."""
    text = ""
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = file_bytes.decode(enc).strip()
            break
        except UnicodeDecodeError:
            continue

    return {
        "text": text,
        "pages": [{"page_num": 1, "text": text}],
        "num_pages": 1,
        "file_type": "text",
        "warnings": [],
    }


def ingest_file(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """
    Main entrypoint for document ingestion.
    Dispatches to appropriate parser based on file extension.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        res = extract_from_pdf(file_bytes)
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
        res = extract_from_image(file_bytes)
    elif ext in (".docx", ".doc"):
        res = extract_from_docx(file_bytes)
    elif ext in (".txt", ".md", ".json", ".csv"):
        res = extract_from_text(file_bytes)
    else:
        # Default fallback: try text decode
        res = extract_from_text(file_bytes)

    # Normalize whitespace
    raw_text = res.get("text", "")
    res["char_count"] = len(raw_text)
    res["word_count"] = len(raw_text.split())
    res["filename"] = filename

    return res


# ---------------------------------------------------------------------------
# ENTITY COLOR MAP for highlight rendering (RGBA tuples)
# ---------------------------------------------------------------------------
_ENTITY_RGBA = {
    "PER":  (59,  130, 246, 100),   # blue
    "LOC":  (34,  197, 94,  100),   # green
    "ORG":  (245, 158, 11,  100),   # amber
    "MISC": (168, 85,  247, 100),   # purple
}
_ENTITY_BORDER = {
    "PER":  (29,  78,  216, 220),
    "LOC":  (21,  128, 61,  220),
    "ORG":  (146, 64,  14,  220),
    "MISC": (107, 33,  168, 220),
}
_ENTITY_LABEL_BG = {
    "PER":  (29,  78,  216),
    "LOC":  (21,  128, 61),
    "ORG":  (146, 64,  14),
    "MISC": (107, 33,  168),
}


def render_pdf_with_highlights(
    file_bytes: bytes,
    entities: list[dict],
    dpi: int = 150,
) -> list[dict]:
    """
    Render each page of a PDF as a PNG image with entity text highlighted.

    For every detected entity, the function searches for the entity text on the
    corresponding PDF page using fitz.Page.search_for(), then draws a semi-
    transparent filled rectangle plus a solid border in the entity-type colour.
    A small label badge (e.g. "PER") is drawn at the top-left corner of each box.

    Args:
        file_bytes: Raw PDF bytes.
        entities:   List of entity dicts from the NER predictor.
                    Each must contain at least: 'text' (str) and 'type' (str).
                    Optionally 'page_num' (1-indexed int) if known; otherwise the
                    function searches across ALL pages.
        dpi:        Render resolution (default 150 dpi for a good screen/print balance).

    Returns:
        List of dicts, one per PDF page that was rendered:
        [
            {
                'page_num': int,          # 1-indexed
                'image_bytes': bytes,     # PNG bytes of the rendered page
                'hit_count': int,         # number of entity highlight boxes drawn
            },
            ...
        ]
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed. Run: pip install pymupdf")

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    zoom = dpi / 72.0  # fitz default is 72 dpi
    mat = fitz.Matrix(zoom, zoom)

    # Build a lookup: page_num (0-indexed) -> list[entity]
    # We try entity['page_num'] if present, otherwise broadcast to all pages.
    page_entities: dict[int, list[dict]] = {i: [] for i in range(len(doc))}
    for ent in entities:
        if not ent.get("text", "").strip():
            continue
        if "page_num" in ent:
            p_idx = int(ent["page_num"]) - 1
            if 0 <= p_idx < len(doc):
                page_entities[p_idx].append(ent)
        else:
            for p_idx in range(len(doc)):
                page_entities[p_idx].append(ent)

    results = []
    for p_idx, page in enumerate(doc):
        ents_for_page = page_entities[p_idx]
        hit_count = 0

        # Annotate using fitz shapes
        shape = page.new_shape()
        for ent in ents_for_page:
            ent_text = ent["text"].strip()
            etype = ent.get("type", "MISC")

            # fitz color is (r,g,b) in 0-1 range
            rgba = _ENTITY_RGBA.get(etype, (120, 120, 120, 80))
            border = _ENTITY_BORDER.get(etype, (60, 60, 60, 200))
            fill_color = (rgba[0] / 255, rgba[1] / 255, rgba[2] / 255)
            stroke_color = (border[0] / 255, border[1] / 255, border[2] / 255)

            # Search for all occurrences of the entity text on this page
            quads = page.search_for(ent_text, quads=True)
            for quad in quads:
                rect = quad.rect
                # Draw filled semi-transparent rect
                shape.draw_rect(rect)
                shape.finish(
                    fill=fill_color,
                    color=stroke_color,
                    fill_opacity=0.35,
                    width=1.5,
                )
                # Draw label badge in top-left corner
                badge_rect = fitz.Rect(
                    rect.x0, rect.y0 - 11,
                    rect.x0 + len(etype) * 6 + 6, rect.y0
                )
                shape.draw_rect(badge_rect)
                shape.finish(
                    fill=fill_color,
                    color=stroke_color,
                    fill_opacity=0.85,
                    width=0,
                )
                # Insert label text
                page.insert_text(
                    fitz.Point(rect.x0 + 2, rect.y0 - 2),
                    etype,
                    fontsize=7,
                    color=stroke_color,
                )
                hit_count += 1

        shape.commit()

        # Render to PNG
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        results.append({
            "page_num": p_idx + 1,
            "image_bytes": png_bytes,
            "hit_count": hit_count,
        })

    doc.close()
    return results


def render_image_with_highlights(
    file_bytes: bytes,
    entities: list[dict],
) -> bytes:
    """
    Annotate an OCR'd image with entity highlight boxes.

    Runs EasyOCR in 'detail' mode to get per-word bounding polygons, then
    matches each recognised word against the entity list and draws a coloured
    semi-transparent rectangle + border + label using PIL.

    Args:
        file_bytes: Raw image bytes (PNG/JPG/etc.).
        entities:   List of entity dicts from the NER predictor.

    Returns:
        PNG bytes of the annotated image.
    """
    if PILImage is None:
        raise RuntimeError("Pillow is not installed. Run: pip install Pillow")

    # Open the source image with PIL (RGBA for compositing)
    base_img = PILImage.open(io.BytesIO(file_bytes)).convert("RGBA")
    overlay = PILImage.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Build a fast lookup: lowercase word -> entity type
    word_to_type: dict[str, str] = {}
    for ent in entities:
        for word in ent.get("text", "").strip().split():
            word_to_type[word.lower()] = ent.get("type", "MISC")

    # Run EasyOCR with detail=1 to get bounding boxes
    reader = get_ocr_reader("indic")
    if reader is None:
        # Cannot do OCR — just return original image as PNG
        buf = io.BytesIO()
        base_img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    ocr_results = reader.readtext(file_bytes, detail=1)
    # ocr_results: list of (bbox, text, confidence)
    # bbox: [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]

    for bbox, word_text, _conf in ocr_results:
        word_clean = word_text.strip().lower()
        # Check if any entity text contains this word
        matched_type = None
        for ent in entities:
            ent_words = [w.lower() for w in ent["text"].strip().split()]
            if word_clean in ent_words or word_clean in ent["text"].lower():
                matched_type = ent.get("type", "MISC")
                break

        if matched_type is None:
            continue

        rgba = _ENTITY_RGBA.get(matched_type, (120, 120, 120, 80))
        border = _ENTITY_BORDER.get(matched_type, (60, 60, 60, 200))
        label_bg = _ENTITY_LABEL_BG.get(matched_type, (60, 60, 60))

        # Compute axis-aligned bounding box from the polygon
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

        # Draw filled semi-transparent highlight
        draw.rectangle([x0, y0, x1, y1], fill=(*rgba[:3], 90), outline=(*border[:3], 220), width=2)

        # Draw label badge above the box
        label = matched_type
        badge_x0, badge_y0 = x0, max(0, y0 - 14)
        badge_x1, badge_y1 = badge_x0 + len(label) * 7 + 4, y0
        draw.rectangle([badge_x0, badge_y0, badge_x1, badge_y1], fill=(*label_bg, 220))
        draw.text((badge_x0 + 2, badge_y0), label, fill=(255, 255, 255, 255))

    # Composite overlay onto base
    combined = PILImage.alpha_composite(base_img, overlay).convert("RGB")
    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    return buf.getvalue()
