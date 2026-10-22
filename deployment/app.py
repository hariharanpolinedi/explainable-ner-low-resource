"""
Phase 15: Streamlit Interactive Explanation & Inference Demo.

Features:
- Multi-script language selection (Konkani, Assamese, Sinhala)
- Interactive text input with curated preset examples from gold datasets
- Color-coded entity span highlighting (PER, LOC, ORG, MISC)
- Visual Gradient × Attention attribution heatmaps per entity
- Top context tokens driving model predictions
- GPU/CPU device status and verified benchmark telemetry
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import streamlit as st
import torch

from deployment.inference import SUPPORTED_LANGUAGES, get_predictor, select_device
from deployment.file_ingestion import ingest_file, render_pdf_with_highlights, render_image_with_highlights
from explainability.heatmap_visualizer import render_html_heatmap

# Page Configuration
st.set_page_config(
    page_title="Explainable Low-Resource NER Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Preset Curated Sentences for Quick Testing
SAMPLE_SENTENCES = {
    "konkani": [
        "ताचो जल्म नावेली साश्ट गोंय हांगा जालो .",
        "‘ अनार्य ’ ( गोवा कोंकणी अकादमी प्रकाशन , पणजी - गोंय - 2010 ) , उजवाडाक आयला .",
        "जाग , राष्ट्रमत आनी भांगरभूंय हीं कोंकणी नेमाळीं आनी खबरांपत्रां .",
        "तातूंत गोंय कला अकादेमी , साहित्य अकादेमी हांचो आसपाव आसा .",
    ],
    "assamese": [
        "১৯২১ চনত মোহনদাস গান্ধী ভাৰতীয় কংগ্ৰেছ দলৰ কাৰ্যবাহী সম্পাদক নিযুক্ত হয় ।",
        "অসমৰ ৰাজধানী দিছপুৰ গুৱাহাটী মহানগৰৰ এটা অংশ ।",
        "ড° ভূপেন হাজৰিকাক ভাৰত চৰকাৰে ভাৰত ৰত্ন সন্মানেৰে সন্মানিত কৰে ।",
        "কাজিৰঙা ৰাষ্ট্ৰীয় উদ্যান অসমৰ গৌৰৱ আৰু ইয়াৰ এশিঙীয়া গঁড় বিশ্ববিখ্যাত ।",
    ],
    "sinhala": [
        "ශ්‍රී ලංකා ජනාධිපති කොළඹ නගරයේ සංචාරය කළේය .",
        "මහින්ද රාජපක්ෂ මහතා අරලියගහ මැදුරේදී විශේෂ සාකච්ඡාවක් පැවැත්වීය .",
        "මහනුවර ශ්‍රී දළදා මාළිගාව බෞද්ධ ජනතාවගේ පූජනීය ස්ථානයකි .",
        "ලංකා විදුලිබල මණ්ඩලය නව ව්‍යාපෘතියක් ආරම්භ කිරීමට සැලසුම් කරයි .",
    ],
    "kokborok": [
        "Maharaja Bir Bikram Kishore Manikya Tripura-ni Agartala-o Ujjayanta Palace snammani .",
        "Tabuk , Tripura Tribal Areas Autonomous District Council-ni pande wngkha Khumulwng .",
        "Kokborok Day romwi January 19 , 1979 bisi-o chengmani .",
        "Pradyot Bikram Manikya Debbarma , Tipra Motha-ni padhai , New Delhi-o kothoma sanlaikha .",
        "India-ni borokrog April tal-o Buisu twi rwng kthar khwlai-o .",
        "Agartala ni order marwsinai ta hing khe .",
        "Where is Mohan Bo phaiwi de tong ?",
    ],
}

LANGUAGE_LABELS = {
    "konkani": "Konkani (कोंकणी - Devanagari)",
    "assamese": "Assamese (অসমীয়া - Eastern Indic)",
    "sinhala": "Sinhala (සිංහල - Southern Brahmic)",
    "kokborok": "Kokborok (Roman / Latin - Tibeto-Burman)",
}

ENTITY_COLORS = {
    "PER": {"bg": "#dbeafe", "text": "#1e40af", "border": "#93c5fd", "label": "Person"},
    "LOC": {"bg": "#dcfce7", "text": "#166534", "border": "#86efac", "label": "Location"},
    "ORG": {"bg": "#fef3c7", "text": "#92400e", "border": "#fde68a", "label": "Organization"},
    "MISC": {"bg": "#f3e8ff", "text": "#6b21a8", "border": "#d8b4fe", "label": "Miscellaneous"},
}

BENCHMARK_SCORES = {
    "konkani": {
        "f1": "96.27%",
        "acc": "99.47%",
        "status": "MET (+6.27 pp)",
        "note": "Cycle 2 Verified (current_best_konkani.pt)",
    },
    "assamese": {
        "f1": "97.87%",
        "acc": "99.64%",
        "status": "MET (+7.87 pp)",
        "note": "Cycle 2 Verified (current_best_assamese.pt, 100% Recall)",
    },
    "sinhala": {
        "f1": "97.76%",
        "acc": "99.70%",
        "status": "MET (+7.76 pp)",
        "note": "Cycle 1 Verified (improved_sinhala_v2.pt)",
    },
    "kokborok": {
        "f1": "82.35%",
        "acc": "99.55%",
        "status": "FINAL BEST (+13.38 pp over Phase 13)",
        "note": "Authoritative Final Model (kokborok_exp4_best.pt: P=77.78%, R=87.50%, F1=82.35%). Certified in Springer Final Manuscript.",
    },
}


def render_highlighted_sentence(tokens: list[str], entities: list[dict[str, Any]]) -> str:
    """
    Render sentence with color-coded entity span highlights.
    """
    # Create map of token index -> entity
    token_to_ent: dict[int, tuple[dict[str, Any], bool, bool]] = {}
    for ent in entities:
        s = ent["start_token"]
        e = ent["end_token"]
        for idx in range(s, e):
            token_to_ent[idx] = (ent, idx == s, idx == e - 1)

    spans_html = []
    i = 0
    while i < len(tokens):
        if i in token_to_ent:
            ent, is_start, is_end = token_to_ent[i]
            etype = ent["type"]
            style = ENTITY_COLORS.get(etype, {"bg": "#e2e8f0", "text": "#334155", "border": "#cbd5e1"})
            
            # Collect full entity text
            ent_toks = tokens[ent["start_token"] : ent["end_token"]]
            safe_text = html.escape(" ".join(ent_toks))
            
            badge_html = (
                f'<span style="background-color: {style["bg"]}; color: {style["text"]}; '
                f'border: 1px solid {style["border"]}; border-radius: 6px; padding: 2px 8px; '
                f'margin: 0 4px; display: inline-block; font-weight: 600;">'
                f'{safe_text} '
                f'<span style="font-size: 0.72em; opacity: 0.85; text-transform: uppercase; '
                f'background: rgba(0,0,0,0.06); padding: 1px 4px; border-radius: 3px; margin-left: 4px;">'
                f'{etype}</span>'
                f'</span>'
            )
            spans_html.append(badge_html)
            i = ent["end_token"]
        else:
            spans_html.append(f'<span style="margin: 0 2px;">{html.escape(tokens[i])}</span>')
            i += 1

    return f'<div style="font-size: 1.25em; line-height: 2.2; padding: 18px; border-radius: 8px; background: #ffffff; border: 1px solid #e2e8f0;">{" ".join(spans_html)}</div>'


def main() -> None:
    # Sidebar
    st.sidebar.title("Configuration & Telemetry")
    lang_choice = st.sidebar.selectbox(
        "Select Language",
        options=list(SUPPORTED_LANGUAGES),
        format_func=lambda k: LANGUAGE_LABELS[k],
    )

    device = select_device()
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    st.sidebar.markdown(f"**Hardware Device**: `{device}` ({device_name})")

    # Benchmark display
    bench = BENCHMARK_SCORES[lang_choice]
    st.sidebar.markdown("---")
    st.sidebar.subheader("Verified Gold-Test Benchmark")
    st.sidebar.metric("Entity-Level F1 (Primary)", bench["f1"], delta=bench["status"])
    st.sidebar.metric("Token Accuracy (Secondary)", bench["acc"])
    st.sidebar.caption(
        f"**Audit Status**: {bench['note']}. Evaluated against cryptographically locked gold test split under strict single-run protocol."
    )
    if lang_choice == "kokborok":
        st.sidebar.warning(
            "⚠️ **Small Sample Size Disclosure**: Kokborok's gold test split contains only 16 entities total across 200 sentences "
            "(PER: 2, LOC: 10, ORG: 1, MISC: 3). Unlike Konkani, Assamese, and Sinhala (124–138 entities each), this evaluation is "
            "statistically under-powered and serves as an exploratory low-resource feasibility baseline."
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Architecture")
    st.sidebar.markdown(
        "- **Embedding**: Word (100d) + Char-CNN (90d) + Lang (16d)\n"
        "- **Encoder**: 2-Layer BiLSTM (256d)\n"
        "- **Attention**: Multi-Head Self-Attention (4 heads)\n"
        "- **Decoder**: Linear-Chain CRF (Viterbi)\n"
        "- **Attribution**: Gradient × Attention"
    )

    # Main Panel
    st.title("Explainable Named Entity Recognition (NER)")
    st.markdown(
        "**Autonomous Low-Resource NER for Konkani, Assamese, Sinhala, and Kokborok** using custom "
        "Char-CNN + BiLSTM + Self-Attention with **Gradient × Attention attribution explanations**."
    )

    tab_manual, tab_file = st.tabs([
        "✍️ Direct Text Input & Examples",
        "📁 Document & Photo Ingestion (PDF, Image, DOCX, TXT)"
    ])

    with tab_manual:
        # Preset Sample Selection
        samples = SAMPLE_SENTENCES[lang_choice]
        col_sel, col_btn = st.columns([4, 1])
        with col_sel:
            preset = st.selectbox("Choose an Example Sentence or Type Below:", ["(Custom Input)"] + samples)

        initial_text = "" if preset == "(Custom Input)" else preset
        user_sentence = st.text_area(
            "Input Sentence",
            value=initial_text,
            height=100,
            placeholder="Enter a sentence in the native script...",
            key="manual_sentence_input"
        )

        run_analysis = st.button("Run Explainable NER", type="primary", use_container_width=True, key="btn_run_manual")

        if run_analysis or (user_sentence and preset != "(Custom Input)"):
            if not user_sentence.strip():
                st.warning("Please enter a non-empty sentence to analyze.")
            else:
                _process_and_display_results(user_sentence, lang_choice, is_doc=False)

    with tab_file:
        st.markdown("### Upload Any Document or Image for Multilingual NER")
        st.caption("Upload **PDF documents**, **Photos/Images (OCR)**, **Word files (.docx)**, or **Plain text (.txt)**. The engine automatically extracts the text and runs entity tagging.")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "docx", "txt", "md"],
            help="Supports PDF, Images (PNG/JPG/WEBP), Word documents, and text files.",
            key="file_uploader_widget"
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            with st.spinner(f"Ingesting and extracting text from {uploaded_file.name}..."):
                try:
                    ingest_res = ingest_file(file_bytes, uploaded_file.name)
                except Exception as e:
                    st.error(f"Error extracting text from file: {e}")
                    ingest_res = None

            if ingest_res:
                col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
                col_meta1.metric("File Type", ingest_res["file_type"].upper())
                col_meta2.metric("Total Words", f"{ingest_res['word_count']:,}")
                col_meta3.metric("Total Characters", f"{ingest_res['char_count']:,}")
                col_meta4.metric("Pages", ingest_res.get("num_pages", 1))

                # If image, show preview
                if ingest_res["file_type"] == "image":
                    with st.expander("🖼️ View Uploaded Photo / Image", expanded=True):
                        st.image(file_bytes, caption=uploaded_file.name, use_container_width=True)

                if ingest_res.get("warnings"):
                    for w in ingest_res["warnings"]:
                        st.warning(w)

                extracted_text = st.text_area(
                    "Extracted Text (Review & Edit if Needed)",
                    value=ingest_res["text"],
                    height=200,
                    key="extracted_file_text_area",
                    help="You can edit, fix typos, or add text before running the NER model."
                )

                col_run_doc, col_clear = st.columns([3, 1])
                with col_run_doc:
                    run_doc_analysis = st.button(
                        f"🔍 Extract Entities from Document ({LANGUAGE_LABELS[lang_choice].split()[0]})",
                        type="primary",
                        use_container_width=True,
                        key="btn_run_doc"
                    )

                if run_doc_analysis:
                    if not extracted_text.strip():
                        st.warning("The extracted text is empty. Please enter or upload valid text.")
                    else:
                        _process_and_display_results(
                            extracted_text,
                            lang_choice,
                            is_doc=True,
                            filename=uploaded_file.name,
                            file_bytes=file_bytes,
                            file_type=ingest_res["file_type"],
                        )


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences respecting Indic dandas and Latin punctuation."""
    # Split on dandas, periods, question marks, newlines
    raw_sents = re.split(r'([।\.\?\!\n]+)', text)
    sents = []
    curr = ""
    for piece in raw_sents:
        curr += piece
        if re.search(r'[।\.\?\!\n]', piece) or len(curr) > 150:
            c_strip = curr.strip()
            if c_strip:
                sents.append(c_strip)
            curr = ""
    if curr.strip():
        sents.append(curr.strip())
    return [s for s in sents if len(s.strip()) > 1] or [text.strip()]


def _process_and_display_results(
    text: str,
    lang_choice: str,
    is_doc: bool = False,
    filename: str | None = None,
    file_bytes: bytes | None = None,
    file_type: str | None = None,
) -> None:
    """Core inference and display pipeline for single sentences or documents."""
    sentences = _split_into_sentences(text) if is_doc else [text]

    all_entities = []
    all_tokens = []
    rendered_blocks = []

    with st.spinner(f"Analyzing {len(sentences)} sentence(s) with {lang_choice.capitalize()} NER model on GPU..."):
        try:
            predictor = get_predictor(lang_choice)
            for idx, sent in enumerate(sentences):
                # Include attribution only for single sentence or first 3 sentences of doc to preserve speed
                inc_attr = not is_doc or (idx < 3)
                res = predictor.predict(sent, include_attribution=inc_attr)
                
                s_tokens = res["tokens"]
                s_entities = res["entities"]
                all_tokens.extend(s_tokens)
                
                # Adjust entity spans for document aggregation
                for ent in s_entities:
                    ent["sentence_idx"] = idx + 1
                    ent["sentence_context"] = sent
                    all_entities.append(ent)
                
                if s_entities:
                    rendered_blocks.append(render_highlighted_sentence(s_tokens, s_entities))
                else:
                    rendered_blocks.append(f'<div style="font-size: 1.1em; line-height: 1.8; padding: 10px; border-radius: 6px; background: #f8fafc; border: 1px solid #e2e8f0; margin-bottom: 8px;">{html.escape(sent)}</div>')
        except Exception as e:
            st.error(f"Inference error: {e}")
            return

    st.subheader("1. Entity Extraction & Tagging")

    # Document Level Summary Cards
    counts = {"PER": 0, "LOC": 0, "ORG": 0, "MISC": 0}
    for ent in all_entities:
        counts[ent["type"]] = counts.get(ent["type"], 0) + 1

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Entities", len(all_entities))
    c2.metric("👤 Person (PER)", counts["PER"])
    c3.metric("📍 Location (LOC)", counts["LOC"])
    c4.metric("🏛️ Organization (ORG)", counts["ORG"])
    c5.metric("🏷️ Miscellaneous (MISC)", counts["MISC"])

    # Highlighted Text
    st.markdown("#### Document Text with Entity Tags")
    st.markdown("<br>".join(rendered_blocks), unsafe_allow_html=True)

    # Entities Structured Table
    if all_entities:
        st.markdown("---")
        st.markdown("#### Detected Entities Summary Table")
        
        # Deduplicate and count occurrences
        from collections import Counter
        ent_counts = Counter((e["text"].strip(), e["type"]) for e in all_entities)
        
        table_data = []
        for (ent_text, ent_type), count in ent_counts.most_common():
            table_data.append({
                "Entity Text": ent_text,
                "Type": ent_type,
                "Category": ENTITY_COLORS.get(ent_type, {}).get("label", ent_type),
                "Occurrences": count
            })
        
        st.dataframe(table_data, use_container_width=True)

        # Download Buttons
        col_dl_json, col_dl_csv = st.columns(2)
        import json
        import io
        import csv

        # Prepare JSON download
        json_str = json.dumps({
            "language": lang_choice,
            "filename": filename or "manual_input",
            "total_entities": len(all_entities),
            "entities": all_entities
        }, ensure_ascii=False, indent=2)

        with col_dl_json:
            st.download_button(
                label="📥 Download Entities (JSON)",
                data=json_str,
                file_name=f"{Path(filename or 'ner_results').stem}_entities.json",
                mime="application/json",
                use_container_width=True
            )

        # Prepare CSV download
        csv_buffer = io.StringIO()
        csv_writer = csv.DictWriter(csv_buffer, fieldnames=["Entity Text", "Type", "Category", "Occurrences"])
        csv_writer.writeheader()
        csv_writer.writerows(table_data)
        csv_data = csv_buffer.getvalue()

        with col_dl_csv:
            st.download_button(
                label="📥 Download Entities (CSV)",
                data=csv_data,
                file_name=f"{Path(filename or 'ner_results').stem}_entities.csv",
                mime="text/csv",
                use_container_width=True
            )

    # -----------------------------------------------------------------------
    # Visual PDF / Image Highlights
    # -----------------------------------------------------------------------
    if all_entities and file_bytes and file_type:
        st.markdown("---")
        st.subheader("📸 Visual Entity Highlights")
        st.caption(
            "The source document is rendered below with detected entities **highlighted directly on the page**. "
            "Each colour represents a different entity type: "
            "🟦 Person (PER) · 🟩 Location (LOC) · 🟧 Organization (ORG) · 🟪 Misc (MISC)"
        )

        if file_type == "pdf":
            with st.spinner("Rendering annotated PDF pages..."):
                try:
                    page_renders = render_pdf_with_highlights(file_bytes, all_entities)
                    pages_with_hits = [p for p in page_renders if p["hit_count"] > 0]
                    if not pages_with_hits:
                        # Fall back to showing all pages
                        pages_with_hits = page_renders

                    for pr in pages_with_hits:
                        st.markdown(
                            f"**Page {pr['page_num']}** — {pr['hit_count']} entity highlight(s)"
                        )
                        st.image(
                            pr["image_bytes"],
                            caption=f"Page {pr['page_num']} with NER highlights",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.warning(f"Could not render PDF highlights: {e}")

        elif file_type == "image":
            with st.spinner("Rendering annotated image with entity highlights..."):
                try:
                    annotated_bytes = render_image_with_highlights(file_bytes, all_entities)
                    st.image(
                        annotated_bytes,
                        caption="Image with NER entity highlights",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"Could not render image highlights: {e}")

    # Attribution Explanations (Gradient x Attention)
    entities_with_attr = [e for e in all_entities if e.get("attributions")]
    if entities_with_attr:
        st.markdown("---")
        st.subheader("2. Attention Attribution Explanations (Gradient × Attention)")
        st.markdown(
            "These heatmaps reveal **which contextual tokens directly influenced the model's decision** "
            "to predict each entity span, computed via output-to-attention gradient attribution:"
        )

        for i, ent in enumerate(entities_with_attr[:6], 1):
            st.markdown(f"#### Entity #{i}: `{ent['text']}` ({ent['type']})")
            tokens_to_use = ent["sentence_context"].split()
            html_card = render_html_heatmap(
                tokens=tokens_to_use,
                attributions=ent["attributions"][:len(tokens_to_use)],
                target_token_idx=min(ent["start_token"], len(tokens_to_use) - 1),
                target_tag=f"{ent['type']} (B-{ent['type']})",
                lang=lang_choice,
            )
            st.components.v1.html(html_card, height=180, scrolling=True)

            if "top_attended_tokens" in ent and ent["top_attended_tokens"]:
                st.markdown("**Most Influential Context Words:**")
                tags_md = " &nbsp;|&nbsp; ".join([
                    f"**{item['token']}** (score: `{item['score']:.4f}`)"
                    for item in ent["top_attended_tokens"][:4]
                ])
                st.markdown(tags_md)



if __name__ == "__main__":
    main()
