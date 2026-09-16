"""
Self-Training Curve Tracking and Visualization (Phase 10).

Plots and tabulates dev-F1 progression vs self-training round per language,
documenting the confidence threshold trajectory, pseudo-data accumulation,
rollback events, and stopping criteria decisions.
Outputs:
- reports/phase10_selftraining_curve.png
- reports/phase10_selftraining_curve.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore


def plot_selftraining_curve(
    results_by_lang: dict[str, list[dict[str, Any]]],
    output_png_path: Path | str = "reports/phase10_selftraining_curve.png",
) -> Path | None:
    """Plot Dev F1 vs Round per language and save PNG figure."""
    if plt is None:
        print("Matplotlib not available; skipping PNG plot generation.")
        return None

    output_png = Path(output_png_path)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=300)
    colors = {"konkani": "#1f77b4", "assamese": "#2ca02c", "sinhala": "#ff7f0e"}
    markers = {"konkani": "o", "assamese": "s", "sinhala": "^"}

    for lang, history in results_by_lang.items():
        rounds = [h["round"] for h in history]
        f1_scores = [h["dev_f1"] * 100 for h in history]
        c = colors.get(lang, "#333333")
        m = markers.get(lang, "o")

        plt.plot(
            rounds,
            f1_scores,
            label=f"{lang.capitalize()} (Final: {f1_scores[-1]:.2f}%)",
            color=c,
            marker=m,
            linewidth=2.2,
            markersize=7,
        )

        for r, score in zip(rounds, f1_scores):
            plt.annotate(
                f"{score:.2f}%",
                (r, score),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=c,
            )

    plt.title("Iterative Self-Training Trajectory Across Rounds (Phase 10)", fontsize=13, fontweight="bold")
    plt.xlabel("Self-Training Round (0 = Phase 9 Best Baseline)", fontsize=11)
    plt.ylabel("Gold-Dev Entity F1 (%)", fontsize=11)
    plt.xticks(sorted(list({h["round"] for hist in results_by_lang.values() for h in hist})))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    plt.savefig(output_png)
    plt.close()
    print(f"Saved self-training curve figure to {output_png}")
    return output_png


def generate_selftraining_report(
    results_by_lang: dict[str, list[dict[str, Any]]],
    summary_by_lang: dict[str, dict[str, Any]],
    output_md_path: Path | str = "reports/phase10_selftraining_curve.md",
    image_rel_path: str = "phase10_selftraining_curve.png",
) -> Path:
    """Generate comprehensive Phase 10 self-training curve report."""
    output_md = Path(output_md_path)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Phase 10: Iterative Self-Training & Confidence Tightening Report",
        "",
        "**Date**: 2026-09-16  ",
        "**Methodology**: High-confidence CRF pseudo-labeling on remaining unlabeled pool with per-round threshold tightening (+0.02/round) and rollback guard.  ",
        "**Model Family**: In-house from-scratch `NERMultilingualModel` (initialized from Phase 9 best checkpoint).  ",
        "**External Checkpoint / Model Use**: STRICTLY NONE (Zero pretrained components).  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Progression Table",
        "",
        "| Language | Baseline (P9) | Best Round | Best Dev F1 | Delta vs P9 | Total Pseudo Sents Added | Stopping Decision |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for lang, s in sorted(summary_by_lang.items()):
        delta_str = f"+{s['delta_f1']*100:.2f}%" if s['delta_f1'] >= 0 else f"{s['delta_f1']*100:.2f}%"
        lines.append(
            f"| **{lang.capitalize()}** | {s['baseline_f1']*100:.2f}% | Round {s['best_round']} | "
            f"**{s['best_dev_f1']*100:.2f}%** | **{delta_str}** | {s['total_pseudo_volume']:,} | `{s['stopping_reason']}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Round-by-Round Self-Training History",
        "",
    ])

    for lang, history in sorted(results_by_lang.items()):
        lines.extend([
            f"### 2.{sorted(results_by_lang.keys()).index(lang)+1} {lang.capitalize()}",
            "",
            "| Round | Threshold | Pseudo Added | Total Pseudo | Dev F1 | Precision | Recall | Token Acc | Action / Status |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ])
        for h in history:
            thresh_str = f"{h['threshold']:.2f}" if h.get("threshold") is not None else "N/A (Base)"
            lines.append(
                f"| {h['round']} | {thresh_str} | {h['pseudo_added']:,} | {h['cumulative_pseudo']:,} | "
                f"**{h['dev_f1']*100:.2f}%** | {h['precision']*100:.2f}% | {h['recall']*100:.2f}% | "
                f"{h['token_acc']*100:.2f}% | {h['status']} |"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Visual Trajectory Curve",
        "",
        f"![Self-Training Curve]({image_rel_path})",
        "",
        "---",
        "",
        "## 4. Key Implementation Highlights",
        "",
        "1. **CRF Sequence-Confidence Filtering**: Normalized sequence log-probability $\\exp\\left(\\frac{s(x, y^*) - \\log Z(x)}{T}\\right)$ directly quantifies average marginal sequence confidence per token. Threshold strictly tightened by +0.02 each successive round.",
        "2. **Volume Cap Enforcement**: Total cumulative pseudo-labeled sentences restricted to $< 3\\times$ combined gold+silver volume ($< 23,400$ sentences).",
        "3. **Automatic Degradation Rollback**: Any round exhibiting lower dev-F1 than the preceding round is rejected; the checkpoint is immediately rolled back to the prior best checkpoint.",
        "4. **Strict Isolation**: Test set (`data/gold/{lang}/test.conll`) remained 100% untouched and excluded from pseudo-labeling candidates.",
        "",
        "---",
        "**Next Phase**: Phase 11 — Data Augmentation & Class-Imbalance Handling.",
    ])

    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved self-training curve report to {output_md}")
    return output_md
