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

import html
import sys
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
}

LANGUAGE_LABELS = {
    "konkani": "Konkani (कोंकणी - Devanagari)",
    "assamese": "Assamese (অসমীয়া - Eastern Indic)",
    "sinhala": "Sinhala (සිංහල - Southern Brahmic)",
}

ENTITY_COLORS = {
    "PER": {"bg": "#dbeafe", "text": "#1e40af", "border": "#93c5fd", "label": "Person"},
    "LOC": {"bg": "#dcfce7", "text": "#166534", "border": "#86efac", "label": "Location"},
    "ORG": {"bg": "#fef3c7", "text": "#92400e", "border": "#fde68a", "label": "Organization"},
    "MISC": {"bg": "#f3e8ff", "text": "#6b21a8", "border": "#d8b4fe", "label": "Miscellaneous"},
}

BENCHMARK_SCORES = {
    "konkani": {"f1": "96.27%", "acc": "99.47%", "status": "MET (+6.27 pp)"},
    "assamese": {"f1": "97.87%", "acc": "99.64%", "status": "MET (+7.87 pp)"},
    "sinhala": {"f1": "97.76%", "acc": "99.70%", "status": "MET (+7.76 pp)"},
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
    st.sidebar.markdown(
        "*Evaluated against locked test split in Phase 14 with strict zero-peeking discipline.*"
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
        "**Autonomous Low-Resource NER for Konkani, Assamese, and Sinhala** using custom "
        "Char-CNN + BiLSTM + Self-Attention with **Gradient × Attention attribution explanations**."
    )

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
    )

    run_analysis = st.button("Run Explainable NER", type="primary", use_container_width=True)

    if run_analysis or (user_sentence and preset != "(Custom Input)"):
        if not user_sentence.strip():
            st.warning("Please enter a non-empty sentence to analyze.")
            return

        with st.spinner(f"Running inference and attribution for {lang_choice.capitalize()}..."):
            try:
                predictor = get_predictor(lang_choice)
                result = predictor.predict(user_sentence, include_attribution=True)
            except Exception as e:
                st.error(f"Inference error: {e}")
                return

        tokens = result["tokens"]
        entities = result["entities"]

        st.subheader("1. Entity Extraction & Tagging")
        if entities:
            st.markdown(render_highlighted_sentence(tokens, entities), unsafe_allow_html=True)
            
            # Entities Table
            st.markdown("##### Detected Entities Summary")
            cols = st.columns(min(len(entities), 4))
            for i, ent in enumerate(entities):
                with cols[i % len(cols)]:
                    st.info(f"**{ent['text']}**\n- Type: `{ent['type']}`\n- Span: `tokens {ent['start_token']}:{ent['end_token']}`")
        else:
            st.info("No named entities detected in this sentence (all tokens classified as 'O').")
            st.write(f"**Tokenized sentence**: `{' '.join(tokens)}`")

        # Attribution Heatmaps
        if entities:
            st.markdown("---")
            st.subheader("2. Attention Attribution Explanations (Gradient × Attention)")
            st.markdown(
                "These heatmaps reveal **which contextual tokens directly influenced the model's decision** "
                "to predict each entity span, computed via output-to-attention gradient attribution:"
            )

            for i, ent in enumerate(entities, 1):
                st.markdown(f"#### Entity #{i}: `{ent['text']}` ({ent['type']})")
                if "attributions" in ent and ent["attributions"]:
                    html_card = render_html_heatmap(
                        tokens=tokens,
                        attributions=ent["attributions"],
                        target_token_idx=ent["start_token"],
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
                else:
                    st.write("*(Attribution not available for this span)*")

        # Raw Token Inspector
        with st.expander("Show Raw Token-Level Debugging Data"):
            st.json({
                "tokens": tokens,
                "tags": result["tags"],
                "entities": entities,
            })


if __name__ == "__main__":
    main()
