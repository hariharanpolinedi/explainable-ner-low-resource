"""
Attention Attribution Heatmap Visualizer (Phase 13).

Renders color-coded per-token attribution overlays for predicted entity spans
using the empirically winning attribution method (Gradient × Attention).
Generates an interactive HTML and Markdown heatmap gallery in:
    reports/phase13_heatmap_gallery/
covering sample predictions per language (Konkani, Assamese, Sinhala) across all
entity categories (PER, LOC, ORG, MISC).
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from explainability.attention_extractor import AttentionExtractor
from explainability.erasure_faithfulness_test import load_dev_sentences_with_entities
from explainability.gradient_attention_attribution import GradientAttentionAttributor
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import LANG_TO_ID, load_or_create_multilingual_vocabs


def render_html_heatmap(
    tokens: Sequence[str],
    attributions: Sequence[float],
    target_token_idx: int,
    target_tag: str,
    lang: str,
) -> str:
    """
    Render a single sentence attribution heatmap as styled HTML.
    Highlights tokens based on attribution intensity.
    """
    max_attr = max(attributions) if attributions else 1.0
    if max_attr <= 1e-9:
        max_attr = 1.0

    token_spans = []
    for i, tok in enumerate(tokens):
        score = attributions[i]
        norm_score = max(0.0, min(1.0, score / max_attr))
        safe_tok = html.escape(tok)

        is_target = (i == target_token_idx)
        border_style = "border: 2px solid #e11d48; font-weight: bold;" if is_target else "border: 1px solid #e2e8f0;"
        # Red-orange highlight proportional to attribution
        bg_color = f"rgba(239, 68, 68, {norm_score * 0.75 + 0.05:.2f})"

        span_html = (
            f'<span style="display: inline-block; margin: 3px; padding: 4px 8px; border-radius: 4px; '
            f'background-color: {bg_color}; {border_style}" title="Token: {safe_tok} | Attribution: {score:.4f}">'
            f'{safe_tok}'
            f'<sub style="font-size: 0.65em; color: #475569; margin-left: 2px;">{score:.2f}</sub>'
            f'</span>'
        )
        token_spans.append(span_html)

    full_html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 16px 0; padding: 16px; border: 1px solid #cbd5e1; border-radius: 8px; background: #ffffff;">
        <div style="margin-bottom: 8px; font-size: 0.9em; color: #64748b;">
            <strong>Language:</strong> {lang.capitalize()} &nbsp;|&nbsp;
            <strong>Predicted Entity:</strong> <span style="background: #fee2e2; color: #991b1b; padding: 2px 6px; border-radius: 4px; font-weight: 600;">{target_tag}</span> &nbsp;|&nbsp;
            <strong>Target Token:</strong> <span style="font-weight: 600;">{html.escape(tokens[target_token_idx])}</span> (pos {target_token_idx})
        </div>
        <div style="line-height: 2.2; font-size: 1.1em;">
            {' '.join(token_spans)}
        </div>
    </div>
    """
    return full_html


def generate_heatmap_gallery(
    gold_dir: Path | str = "data/gold",
    proc_dir: Path | str = "data/processed",
    checkpoints_dir: Path | str = "checkpoints",
    output_dir: Path | str = "reports/phase13_heatmap_gallery",
    samples_per_type: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """
    Generate attribution heatmaps across all languages and entity types.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gold_dir = Path(gold_dir)
    proc_dir = Path(proc_dir)
    checkpoints_dir = Path(checkpoints_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = ("konkani", "assamese", "sinhala")
    types = ["PER", "LOC", "ORG", "MISC"]
    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir, languages=languages)

    gallery_manifest: dict[str, list[dict[str, Any]]] = {l: [] for l in languages}

    for lang in languages:
        ckpt_path = checkpoints_dir / f"final_{lang}.pt"
        dev_file = gold_dir / lang / "dev.conll"

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = NERMultilingualModel(
            word_vocab_size=len(word_to_id),
            char_vocab_size=len(char_to_id),
            num_tags=9,
            num_languages=3,
            hidden_dim=128,
            num_heads=4,
            num_attention_blocks=1,
            crf_config="shared",
        ).to(device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        model.eval()

        extractor = AttentionExtractor(
            model=model,
            word_to_id=word_to_id,
            char_to_id=char_to_id,
            lang=lang,
            lang_id=LANG_TO_ID[lang],
            device=device,
        )
        attributor = GradientAttentionAttributor(extractor)

        dev_sents = load_dev_sentences_with_entities(dev_file)
        type_collected: dict[str, int] = {t: 0 for t in types}

        for s_idx, sent_data in enumerate(dev_sents):
            tokens = sent_data["tokens"]
            for ent in sent_data["entities"]:
                tag = ent["tag"]
                etype = tag.split("-")[1]
                if etype in type_collected and type_collected[etype] < samples_per_type:
                    q_idx = ent["token_idx"]
                    attr_res = attributor.attribute_token(tokens, target_token_idx=q_idx)
                    scores = attr_res["attribution_scores"].tolist()

                    html_card = render_html_heatmap(
                        tokens=tokens,
                        attributions=scores,
                        target_token_idx=q_idx,
                        target_tag=tag,
                        lang=lang,
                    )

                    card_id = f"{lang}_{etype.lower()}_{type_collected[etype] + 1}"
                    card_filename = f"{card_id}.html"
                    card_path = output_dir / card_filename
                    with open(card_path, "w", encoding="utf-8") as fp:
                        fp.write(html_card)

                    gallery_manifest[lang].append({
                        "id": card_id,
                        "file": card_filename,
                        "type": etype,
                        "tag": tag,
                        "target_token": ent["token"],
                        "token_idx": q_idx,
                        "tokens": tokens,
                        "scores": scores,
                        "html_snippet": html_card,
                    })

                    type_collected[etype] += 1

    # Generate master gallery index markdown
    _write_gallery_index(output_dir, gallery_manifest)
    return gallery_manifest


def _write_gallery_index(
    output_dir: Path,
    manifest: dict[str, list[dict[str, Any]]],
) -> None:
    index_path = output_dir / "README.md"
    lines = [
        "# Phase 13: Attention Attribution Heatmap Gallery",
        "",
        "**Attribution Method**: **Gradient × Attention** (Empirically Selected as Most Faithful)  ",
        "**Date**: 2026-09-17  ",
        "**Languages**: Konkani, Assamese, Sinhala  ",
        "**Entity Categories**: PER, LOC, ORG, MISC  ",
        "",
        "---",
        "",
        "## Visual Attribution Gallery",
        "",
        "In each example below, token highlights indicate the causal importance (attribution weight) of contextual words towards the predicted entity token (outlined in red border with bold text).",
        "",
    ]

    for lang, items in manifest.items():
        lines.append(f"### {lang.capitalize()}")
        lines.append("")
        for item in items:
            lines.append(f"#### [{item['type']}] Entity: `{item['target_token']}` ({item['tag']})")
            lines.append(f"- **Top Attributed Tokens**: " + ", ".join(
                f"`{tok}` ({score:.3f})"
                for tok, score, _ in sorted(zip(item["tokens"], item["scores"], range(len(item["tokens"]))), key=lambda x: x[1], reverse=True)[:3]
            ))
            lines.append("")
            lines.append(item["html_snippet"])
            lines.append("")

    with open(index_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    print(f"[Gallery Index Saved] {index_path}")


if __name__ == "__main__":
    generate_heatmap_gallery()
