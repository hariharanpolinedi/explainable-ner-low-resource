"""
Phase 14: Confusion Matrix Generator for Token and Entity Tags.

Computes 9x9 BIO tag confusion matrices and entity-level confusion matrices per language
from official Phase 14 gold-test predictions.
Generates publication-quality heatmap plots saved to reports/phase14_confusion_matrices/{lang}.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.seqeval_metrics import extract_entities
from training.train_charcnn import TAG_TO_ID

TAG_LIST = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
ENTITY_TYPES = ["PER", "LOC", "ORG", "MISC", "O"]


def compute_tag_confusion_matrix(
    gold_sequences: list[list[str]],
    pred_sequences: list[list[str]],
    tag_list: list[str] = TAG_LIST,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute raw token-level tag confusion matrix.
    Rows = True/Gold Tag, Columns = Predicted Tag.
    """
    n = len(tag_list)
    tag_to_idx = {t: i for i, t in enumerate(tag_list)}
    matrix = np.zeros((n, n), dtype=int)

    for g_seq, p_seq in zip(gold_sequences, pred_sequences):
        min_len = min(len(g_seq), len(p_seq))
        for g_tok, p_tok in zip(g_seq[:min_len], p_seq[:min_len]):
            g_idx = tag_to_idx.get(g_tok, 0)
            p_idx = tag_to_idx.get(p_tok, 0)
            matrix[g_idx, p_idx] += 1

    return matrix, tag_list


def compute_entity_confusion_matrix(
    gold_sequences: list[list[str]],
    pred_sequences: list[list[str]],
    types: list[str] = ENTITY_TYPES,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute span-level entity confusion matrix.
    Rows = Gold entity type (or O for spurious), Columns = Predicted entity type (or O for missed).
    """
    n = len(types)
    type_to_idx = {t: i for i, t in enumerate(types)}
    matrix = np.zeros((n, n), dtype=int)

    for g_seq, p_seq in zip(gold_sequences, pred_sequences):
        gold_spans = extract_entities(g_seq)
        pred_spans = extract_entities(p_seq)

        matched_gold = set()
        matched_pred = set()

        # Check overlapping or exact matches
        for p in pred_spans:
            p_type, p_start, p_end = p
            # Check overlap with any gold span
            overlapping_gold = [
                g for g in gold_spans
                if max(p_start, g[1]) < min(p_end, g[2])
            ]
            if overlapping_gold:
                # Take highest overlap
                best_g = max(overlapping_gold, key=lambda g: min(p_end, g[2]) - max(p_start, g[1]))
                g_type, _, _ = best_g
                matrix[type_to_idx[g_type], type_to_idx[p_type]] += 1
                matched_gold.add(best_g)
                matched_pred.add(p)
            else:
                # Spurious entity (gold is O)
                matrix[type_to_idx["O"], type_to_idx[p_type]] += 1
                matched_pred.add(p)

        # Unmatched gold entities (missed as O)
        for g in gold_spans:
            if g not in matched_gold:
                g_type, _, _ = g
                matrix[type_to_idx[g_type], type_to_idx["O"]] += 1

    return matrix, types


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    normalize: bool = False,
) -> None:
    """
    Render and save confusion matrix as a high-contrast heatmap plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    display_matrix = matrix.astype(float)
    if normalize:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(display_matrix, row_sums, out=np.zeros_like(display_matrix), where=row_sums != 0)

    im = ax.imshow(display_matrix, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="Gold True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate values inside cells
    thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text = f"{val:,}" if not normalize else f"{display_matrix[i, j]:.2f}\n({val})"
            color = "white" if matrix[i, j] > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8, fontweight="medium")

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_all_confusion_matrices(reports_dir: Path | None = None) -> dict[str, Any]:
    """
    Generate confusion matrices and plots for all three languages.
    """
    if reports_dir is None:
        reports_dir = PROJECT_ROOT / "reports"
    img_dir = reports_dir / "phase14_confusion_matrices"
    img_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}

    for lang in ["konkani", "assamese", "sinhala"]:
        pred_file = reports_dir / f"phase14_predictions_{lang}.json"
        if not pred_file.is_file():
            raise FileNotFoundError(
                f"Missing prediction record {pred_file}. Run final_test_eval.py first."
            )

        with open(pred_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        gold_tags = data["gold_tags"]
        pred_tags = data["pred_tags"]

        # Tag-level confusion matrix
        tag_mat, tag_lbls = compute_tag_confusion_matrix(gold_tags, pred_tags)
        plot_path = img_dir / f"{lang}.png"
        plot_confusion_matrix(
            tag_mat,
            tag_lbls,
            title=f"Token-Level Tag Confusion Matrix: {lang.capitalize()} (Gold-Test)",
            output_path=plot_path,
        )

        # Entity-level confusion matrix
        ent_mat, ent_lbls = compute_entity_confusion_matrix(gold_tags, pred_tags)
        ent_plot_path = img_dir / f"{lang}_entity_matrix.png"
        plot_confusion_matrix(
            ent_mat,
            ent_lbls,
            title=f"Span-Level Entity Confusion Matrix: {lang.capitalize()} (Gold-Test)",
            output_path=ent_plot_path,
        )

        summary[lang] = {
            "tag_matrix": tag_mat.tolist(),
            "tag_labels": tag_lbls,
            "entity_matrix": ent_mat.tolist(),
            "entity_labels": ent_lbls,
            "plot_path": str(plot_path),
            "entity_plot_path": str(ent_plot_path),
        }
        print(f"Generated confusion matrices and plots for {lang.capitalize()} -> {plot_path}")

    return summary


if __name__ == "__main__":
    generate_all_confusion_matrices()
