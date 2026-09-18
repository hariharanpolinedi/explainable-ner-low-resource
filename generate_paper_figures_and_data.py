"""
Generate Publication-Grade Figures and Backing Tables Data
for 'Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution'
conforming to Springer Nature standards.
"""

import json
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Configure publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})

FIG_DIR = Path("figures")
DATA_DIR = Path("tables_data")
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------
# 1. EXPORT TABLE DATA (JSON & CSV)
# -------------------------------------------------------------

# Table 1: Raw Corpus
raw_corpus_data = [
    {"language": "Konkani", "script": "Devanagari", "raw_sentences": 112235, "clean_sentences": 51488, "tokens": 4950107, "vocab": 21176},
    {"language": "Assamese", "script": "Eastern Nagari", "raw_sentences": 119265, "clean_sentences": 55002, "tokens": 4376034, "vocab": 25288},
    {"language": "Sinhala", "script": "Sinhala", "raw_sentences": 113296, "clean_sentences": 66954, "tokens": 5799421, "vocab": 30230},
]
with open(DATA_DIR / "table1_raw_corpus.json", "w", encoding="utf-8") as f:
    json.dump(raw_corpus_data, f, indent=2)

# Table 2: Benchmark Splits
benchmark_splits_data = [
    {"language": "Konkani", "split": "Gold Train (Expanded)", "sentences": 850, "tokens": 13710, "entities": 592, "per": 144, "loc": 168, "org": 149, "misc": 131},
    {"language": "Konkani", "split": "Gold Dev (Locked)", "sentences": 200, "tokens": 3173, "entities": 125, "per": 20, "loc": 31, "org": 37, "misc": 37},
    {"language": "Konkani", "split": "Gold Test (Locked)", "sentences": 200, "tokens": 3210, "entities": 124, "per": 20, "loc": 30, "org": 37, "misc": 37},
    {"language": "Assamese", "split": "Gold Train (Expanded)", "sentences": 850, "tokens": 14649, "entities": 679, "per": 102, "loc": 395, "org": 67, "misc": 115},
    {"language": "Assamese", "split": "Gold Dev (Locked)", "sentences": 200, "tokens": 3569, "entities": 139, "per": 18, "loc": 79, "org": 15, "misc": 27},
    {"language": "Assamese", "split": "Gold Test (Locked)", "sentences": 200, "tokens": 3349, "entities": 138, "per": 19, "loc": 79, "org": 14, "misc": 26},
    {"language": "Sinhala", "split": "Gold Train (Expanded)", "sentences": 850, "tokens": 16541, "entities": 641, "per": 178, "loc": 224, "org": 119, "misc": 120},
    {"language": "Sinhala", "split": "Gold Dev (Locked)", "sentences": 200, "tokens": 4032, "entities": 131, "per": 37, "loc": 39, "org": 28, "misc": 27},
    {"language": "Sinhala", "split": "Gold Test (Locked)", "sentences": 200, "tokens": 3996, "entities": 132, "per": 38, "loc": 39, "org": 28, "misc": 27},
]
with open(DATA_DIR / "table2_benchmark_splits.json", "w", encoding="utf-8") as f:
    json.dump(benchmark_splits_data, f, indent=2)

# Table 5: Final NER Performance
final_ner_data = [
    {"language": "Konkani", "precision": 99.15, "recall": 93.55, "f1": 96.27, "token_acc": 99.47, "target_margin": "+6.27 pp"},
    {"language": "Assamese", "precision": 95.83, "recall": 100.00, "f1": 97.87, "token_acc": 99.64, "target_margin": "+7.87 pp"},
    {"language": "Sinhala", "precision": 96.32, "recall": 99.24, "f1": 97.76, "token_acc": 99.70, "target_margin": "+7.76 pp"},
    {"language": "Macro Average", "precision": 97.10, "recall": 97.60, "f1": 97.30, "token_acc": 99.60, "target_margin": "+7.30 pp"},
]
with open(DATA_DIR / "table5_final_ner.json", "w", encoding="utf-8") as f:
    json.dump(final_ner_data, f, indent=2)

# Table 9: Component Ablation
ablation_data = [
    {"condition": "Full Final Pipeline (Locked Ref)", "dev_f1": 98.28, "delta": 0.00, "rank": 0},
    {"condition": "(f) No Data Augmentation", "dev_f1": 97.91, "delta": -0.37, "rank": 1},
    {"condition": "(e) No Self-Training", "dev_f1": 97.91, "delta": -0.37, "rank": 2},
    {"condition": "(d) No Multilingual Joint Warm-Start", "dev_f1": 96.42, "delta": -1.86, "rank": 3},
    {"condition": "(a) No Char-CNN Encoder", "dev_f1": 92.39, "delta": -5.89, "rank": 4},
    {"condition": "(b) No Self-Attention (Matched Full)", "dev_f1": 91.66, "delta": -6.62, "rank": 5},
    {"condition": "(c) No CRF (Softmax Cross-Entropy)", "dev_f1": 91.12, "delta": -7.16, "rank": 6},
    {"condition": "(g) Subword BPE vs Word+Char", "dev_f1": 90.73, "delta": -7.55, "rank": 7},
]
with open(DATA_DIR / "table9_ablations.json", "w", encoding="utf-8") as f:
    json.dump(ablation_data, f, indent=2)

# Table 10: Faithfulness Primary Matrix
faithfulness_data = [
    {"method": "Integrated Gradients", "k1_macro": 0.2449, "k2_macro": 0.3105, "k3_macro": 0.3396, "win_rate": 63.3, "p_val": "< 0.0001"},
    {"method": "Gradient × Input", "k1_macro": 0.2350, "k2_macro": 0.2823, "k3_macro": 0.3057, "win_rate": 62.7, "p_val": "< 0.0001"},
    {"method": "Grad × Attn (Signed ReLU)", "k1_macro": 0.0574, "k2_macro": 0.0841, "k3_macro": 0.1128, "win_rate": 35.0, "p_val": "< 0.01"},
    {"method": "Grad-Only Input", "k1_macro": 0.1365, "k2_macro": 0.1820, "k3_macro": 0.2105, "win_rate": 42.0, "p_val": "< 0.0001"},
    {"method": "Grad-Only Attention", "k1_macro": 0.1067, "k2_macro": 0.1415, "k3_macro": 0.1680, "win_rate": 39.7, "p_val": "< 0.001"},
    {"method": "Uniform Control", "k1_macro": 0.0200, "k2_macro": 0.0280, "k3_macro": 0.0350, "win_rate": 23.7, "p_val": "n.s."},
    {"method": "Random Control", "k1_macro": -0.0049, "k2_macro": -0.0062, "k3_macro": -0.0080, "win_rate": 24.3, "p_val": "n.s."},
    {"method": "Raw Attention (Baseline)", "k1_macro": -0.0279, "k2_macro": -0.0422, "k3_macro": -0.0617, "win_rate": 22.0, "p_val": "ref"},
]
with open(DATA_DIR / "table10_faithfulness.json", "w", encoding="utf-8") as f:
    json.dump(faithfulness_data, f, indent=2)

# Distance Data
distance_data = [
    {"bin": "Dist 0 (Entity)", "raw_attention": 0.14, "input_attribution": 46.83},
    {"bin": "Dist 1 (Immediate)", "raw_attention": 2.10, "input_attribution": 19.69},
    {"bin": "Dist 2", "raw_attention": 6.12, "input_attribution": 11.28},
    {"bin": "Dist 3", "raw_attention": 8.78, "input_attribution": 5.60},
    {"bin": "Dist 4-5", "raw_attention": 21.34, "input_attribution": 6.09},
    {"bin": "Dist > 5 (Distant)", "raw_attention": 61.53, "input_attribution": 10.51},
]
with open(DATA_DIR / "table13_distance.json", "w", encoding="utf-8") as f:
    json.dump(distance_data, f, indent=2)


# -------------------------------------------------------------
# 2. GENERATE ALL 11 PUBLICATION FIGURES
# -------------------------------------------------------------

print("Generating Figure 1: Overall Research Workflow...")
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.axis('off')

boxes = [
    ("Raw Corpus Harvesting\n(Wikipedia & News)\n>170k Raw Sentences", (0.05, 0.65), "#E8F0FE", "#1A73E8"),
    ("Script Normalization &\nPurity Filtering (≥0.90)\n173,444 Clean Sentences", (0.28, 0.65), "#E8F0FE", "#1A73E8"),
    ("Grapheme Tokenization\n& Weak Supervision\n21.6k Silver + 2.5k Gold", (0.51, 0.65), "#E8F0FE", "#1A73E8"),
    ("From-Scratch Neural NER\nWord + Char-CNN +\nBiLSTM + Attn + CRF", (0.74, 0.65), "#E6F4EA", "#137333"),
    ("Multi-Stage Training\nJoint Warm-Start + Self-Train\n+ Targeted Augmentation", (0.05, 0.15), "#E6F4EA", "#137333"),
    ("Single Locked Test Eval\n97.30% Macro Entity F1\nStrict seqeval Boundary", (0.28, 0.15), "#FEF7E0", "#B06000"),
    ("Multi-Tier Attribution\nAttention Layer Routing vs.\nInput Gradient Saliency", (0.51, 0.15), "#FCE8E6", "#C5221F"),
    ("Controlled Faithfulness\nSymmetric Context Erasure\n10k Permutations (p<0.0001)", (0.74, 0.15), "#F3E8FD", "#8430CE"),
]

for text, (x, y), facecol, edgecol in boxes:
    rect = patches.FancyBboxPatch((x, y), 0.20, 0.24, boxstyle="round,pad=0.02,rounding_size=0.03",
                                  facecolor=facecol, edgecolor=edgecol, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 0.10, y + 0.12, text, ha='center', va='center', fontsize=8.5, weight='semibold', color='#202124')

# Connect arrows
arrow_kwargs = dict(arrowprops=dict(arrowstyle="->", color="#5f6368", lw=1.6, mutation_scale=14))
# Row 1
ax.annotate("", xy=(0.28, 0.77), xytext=(0.25, 0.77), **arrow_kwargs)
ax.annotate("", xy=(0.51, 0.77), xytext=(0.48, 0.77), **arrow_kwargs)
ax.annotate("", xy=(0.74, 0.77), xytext=(0.71, 0.77), **arrow_kwargs)
# Row 1 to Row 2
ax.annotate("", xy=(0.84, 0.39), xytext=(0.84, 0.65), **arrow_kwargs)
# Row 2 (right to left or looped)
ax.annotate("", xy=(0.71, 0.27), xytext=(0.74, 0.27), **arrow_kwargs)
ax.annotate("", xy=(0.48, 0.27), xytext=(0.51, 0.27), **arrow_kwargs)
ax.annotate("", xy=(0.25, 0.27), xytext=(0.28, 0.27), **arrow_kwargs)

ax.set_title("Fig. 1  Comprehensive End-to-End Research and Evaluation Pipeline", fontsize=11, weight='bold', pad=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_overall_workflow.png")
fig.savefig(FIG_DIR / "fig1_overall_workflow.pdf")
plt.close(fig)


print("Generating Figure 2: Dataset Construction Pipeline...")
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.axis('off')

steps = [
    ("Raw Data Ingestion", "Wikipedia Dumps & News\nKonkani: 112.2k | Assamese: 119.3k | Sinhala: 113.3k", (0.05, 0.70), "#E8F0FE", "#1A73E8"),
    ("Script & Quality Filtering", "Script Purity ≥ 0.90 | Len: 3-1000 chars | SHA-256 Dedup\nClean: Konkani 51.5k | Assamese 55.0k | Sinhala 67.0k", (0.05, 0.40), "#E8F0FE", "#1A73E8"),
    ("Gold Benchmark (Locked)", "Seed: 600 Train / 200 Dev / 200 Test\nExpanded Train: 850 / lang (Dev & Test Locked)", (0.55, 0.70), "#FEF7E0", "#B06000"),
    ("Weak Supervision (Silver)", "12x Multiplier: 7,200 Silver Sents / lang (21,600 Total)\nGazetteers & Morphological Rule Taggers", (0.55, 0.40), "#E6F4EA", "#137333"),
    ("Semi-Supervised Expansion", "CRF Confidence-Filtered Self-Training (>0.90) + Augmentation\nFinal Unified Training Pool: 36,736 Sentences", (0.25, 0.08), "#F3E8FD", "#8430CE"),
]

for title, desc, (x, y), fcol, ecol in steps:
    width = 0.42 if x != 0.25 else 0.50
    rect = patches.FancyBboxPatch((x, y), width, 0.22, boxstyle="round,pad=0.02,rounding_size=0.03",
                                  facecolor=fcol, edgecolor=ecol, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width/2, y + 0.15, title, ha='center', va='center', fontsize=9.5, weight='bold', color='#202124')
    ax.text(x + width/2, y + 0.07, desc, ha='center', va='center', fontsize=8, color='#3c4043')

ax.annotate("", xy=(0.26, 0.62), xytext=(0.26, 0.70), **arrow_kwargs)
ax.annotate("", xy=(0.55, 0.81), xytext=(0.47, 0.81), **arrow_kwargs)
ax.annotate("", xy=(0.55, 0.51), xytext=(0.47, 0.51), **arrow_kwargs)
ax.annotate("", xy=(0.50, 0.30), xytext=(0.35, 0.40), **arrow_kwargs)
ax.annotate("", xy=(0.50, 0.30), xytext=(0.65, 0.40), **arrow_kwargs)

ax.set_title("Fig. 2  Multi-Source Dataset Construction and Expansion Pipeline", fontsize=11, weight='bold', pad=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_dataset_pipeline.png")
fig.savefig(FIG_DIR / "fig2_dataset_pipeline.pdf")
plt.close(fig)


print("Generating Figure 3: Complete Model Architecture...")
fig, ax = plt.subplots(figsize=(8.5, 6.2))
ax.axis('off')

layers = [
    ("Token Inputs", "Words (w_t), Extended Grapheme Clusters (c_t), Language ID (l)", 0.88, 0.12, "#FFFFFF", "#5F6368"),
    ("Multi-Channel Representation", "Word Emb (100d)  ||  Char-CNN (90d)  ||  Language Emb (16d)\nConcatenated Representation: e_t in R^206 (Dropout 0.3)", 0.72, 0.12, "#E8F0FE", "#1A73E8"),
    ("Bidirectional LSTM Encoder", "2-Layer BiLSTM (Input: 206d, Hidden: 128d/dir, Bidirectional Out: 256d)\nh_t = [h_fwd || h_bwd] (Dropout 0.3)", 0.56, 0.12, "#E6F4EA", "#137333"),
    ("Multi-Head Self-Attention", "4 Heads (d_k=64)  |  Scaled Dot-Product  |  Padding Mask M\nResidual Connection (+)  &  LayerNorm  ->  z_t in R^256", 0.40, 0.12, "#FEF7E0", "#B06000"),
    ("Linear Emission Layer", "Dense Linear Projection W_E in R^256x9  ->  Token Logits P in R^Tx9", 0.24, 0.11, "#FCE8E6", "#C5221F"),
    ("Linear-Chain CRF Decoder", "Transition Matrix T in R^9x9  |  Forward-Backward Partition Z(x)\nExact Viterbi Decoding  ->  Predicted BIO Sequence y_hat", 0.08, 0.12, "#F3E8FD", "#8430CE"),
]

for title, desc, y, h, fcol, ecol in layers:
    rect = patches.FancyBboxPatch((0.15, y), 0.70, h, boxstyle="round,pad=0.015,rounding_size=0.03",
                                  facecolor=fcol, edgecolor=ecol, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.50, y + h*0.65, title, ha='center', va='center', fontsize=10, weight='bold', color='#202124')
    ax.text(0.50, y + h*0.28, desc, ha='center', va='center', fontsize=8.2, color='#3c4043')
    if y > 0.10:
        ax.annotate("", xy=(0.50, y - 0.03), xytext=(0.50, y), **arrow_kwargs)

ax.set_title("Fig. 3  Complete Hybrid Neural Architecture: Char-CNN + BiLSTM + Multi-Head Self-Attention + CRF", fontsize=11, weight='bold', pad=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_model_architecture.png")
fig.savefig(FIG_DIR / "fig3_model_architecture.pdf")
plt.close(fig)


print("Generating Figure 4: Character CNN Architecture...")
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.axis('off')

# Input
rect = patches.FancyBboxPatch((0.05, 0.40), 0.20, 0.35, boxstyle="round,pad=0.02", facecolor="#E8F0FE", edgecolor="#1A73E8", lw=1.5)
ax.add_patch(rect)
ax.text(0.15, 0.62, "Grapheme Cluster\nSequence (c_1..c_L)", ha='center', va='center', fontsize=9, weight='bold')
ax.text(0.15, 0.48, "Char Embeddings\nDim: 50d", ha='center', va='center', fontsize=8)

# Parallel Conv Filters
convs = [
    ("Conv1D (k=3)", "30 Filters\nReLU", 0.70, "#E6F4EA", "#137333"),
    ("Conv1D (k=4)", "30 Filters\nReLU", 0.45, "#E6F4EA", "#137333"),
    ("Conv1D (k=5)", "30 Filters\nReLU", 0.20, "#E6F4EA", "#137333"),
]

for title, desc, y, fcol, ecol in convs:
    rect = patches.FancyBboxPatch((0.36, y), 0.22, 0.18, boxstyle="round,pad=0.015", facecolor=fcol, edgecolor=ecol, lw=1.4)
    ax.add_patch(rect)
    ax.text(0.47, y + 0.11, title, ha='center', va='center', fontsize=9, weight='bold')
    ax.text(0.47, y + 0.05, desc, ha='center', va='center', fontsize=7.5)
    ax.annotate("", xy=(0.36, y + 0.09), xytext=(0.25, 0.57), **arrow_kwargs)

# Max-Pool
rect_pool = patches.FancyBboxPatch((0.66, 0.30), 0.14, 0.45, boxstyle="round,pad=0.02", facecolor="#FEF7E0", edgecolor="#B06000", lw=1.5)
ax.add_patch(rect_pool)
ax.text(0.73, 0.55, "Temporal\nMax-Pooling\nAcross\nPositions", ha='center', va='center', fontsize=8.5, weight='bold')
for _, _, y, _, _ in convs:
    ax.annotate("", xy=(0.66, 0.52), xytext=(0.58, y + 0.09), **arrow_kwargs)

# Output
rect_out = patches.FancyBboxPatch((0.85, 0.35), 0.12, 0.35, boxstyle="round,pad=0.02", facecolor="#F3E8FD", edgecolor="#8430CE", lw=1.5)
ax.add_patch(rect_out)
ax.text(0.91, 0.55, "Subword\nVector\nh_char\n(90d)", ha='center', va='center', fontsize=9, weight='bold')
ax.annotate("", xy=(0.85, 0.52), xytext=(0.80, 0.52), **arrow_kwargs)

ax.set_title("Fig. 4  Multi-Filter Character-Level Convolutional Subword Encoder", fontsize=11, weight='bold', pad=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_char_cnn.png")
fig.savefig(FIG_DIR / "fig4_char_cnn.pdf")
plt.close(fig)


print("Generating Figure 5: Multi-Head Attention Architecture...")
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.axis('off')

# Input H
rect_h = patches.FancyBboxPatch((0.05, 0.42), 0.16, 0.25, boxstyle="round,pad=0.02", facecolor="#E6F4EA", edgecolor="#137333", lw=1.5)
ax.add_patch(rect_h)
ax.text(0.13, 0.55, "BiLSTM Hidden\nStates H in R^Tx256", ha='center', va='center', fontsize=8.5, weight='bold')

# Projections Q, K, V
heads = [
    ("Head 1 (d_k=64)", 0.72),
    ("Head 2 (d_k=64)", 0.52),
    ("Head 3 (d_k=64)", 0.32),
    ("Head 4 (d_k=64)", 0.12),
]

for name, y in heads:
    rect = patches.FancyBboxPatch((0.30, y), 0.22, 0.14, boxstyle="round,pad=0.015", facecolor="#FEF7E0", edgecolor="#B06000", lw=1.3)
    ax.add_patch(rect)
    ax.text(0.41, y + 0.07, f"{name}\nsoftmax(QK^T/√64 + M)V", ha='center', va='center', fontsize=7.5, weight='bold')
    ax.annotate("", xy=(0.30, y + 0.07), xytext=(0.21, 0.55), **arrow_kwargs)

# Concat & Projection
rect_proj = patches.FancyBboxPatch((0.58, 0.35), 0.18, 0.35, boxstyle="round,pad=0.02", facecolor="#E8F0FE", edgecolor="#1A73E8", lw=1.5)
ax.add_patch(rect_proj)
ax.text(0.67, 0.57, "Concat Heads\n& Linear Proj\n(W_O in R^256x256)", ha='center', va='center', fontsize=8.5, weight='bold')
for _, y in heads:
    ax.annotate("", xy=(0.58, 0.52), xytext=(0.52, y + 0.07), **arrow_kwargs)

# Residual + LayerNorm
rect_ln = patches.FancyBboxPatch((0.82, 0.35), 0.15, 0.35, boxstyle="round,pad=0.02", facecolor="#F3E8FD", edgecolor="#8430CE", lw=1.5)
ax.add_patch(rect_ln)
ax.text(0.895, 0.57, "Residual (+)\n&\nLayerNorm\nOut: R^Tx256", ha='center', va='center', fontsize=8.5, weight='bold')
ax.annotate("", xy=(0.82, 0.52), xytext=(0.76, 0.52), **arrow_kwargs)

# Skip connection
ax.annotate("", xy=(0.85, 0.70), xytext=(0.13, 0.67),
            arrowprops=dict(arrowstyle="->", color="#137333", lw=1.4, linestyle="--",
                            connectionstyle="arc3,rad=-0.25"))

ax.set_title("Fig. 5  Scaled Dot-Product Multi-Head Self-Attention Block with Residual Normalization", fontsize=11, weight='bold', pad=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_multihead_attention.png")
fig.savefig(FIG_DIR / "fig5_multihead_attention.pdf")
plt.close(fig)


print("Generating Figure 6: Explainability Framework (Two-Tier Attribution)...")
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.axis('off')

# Tier 1
rect1 = patches.FancyBboxPatch((0.08, 0.54), 0.84, 0.36, boxstyle="round,pad=0.02", facecolor="#FFF8E1", edgecolor="#FFA000", lw=1.5)
ax.add_patch(rect1)
ax.text(0.50, 0.83, "Tier 1: Attention-Layer Explainers (Post-BiLSTM Routing)", ha='center', va='center', fontsize=10, weight='bold', color="#B06000")
ax.text(0.50, 0.69, "• Raw Attention: α_{q→j}  →  Allocates 61.5% mass distantly; broad syntactic broadcast bus (Gap: -0.0279)\n"
                    "• Signed ReLU Grad × Attn: ReLU(α ⊙ ∇_α z)  →  Removes inhibitory noise; positive gap (+0.0574)\n"
                    "• Inherent Limitation: Confounded by pre-attention BiLSTM recurrent state mixing",
        ha='center', va='center', fontsize=8.3, color="#37474F")

# Tier 2
rect2 = patches.FancyBboxPatch((0.08, 0.08), 0.84, 0.36, boxstyle="round,pad=0.02", facecolor="#E8F5E9", edgecolor="#388E3C", lw=1.5)
ax.add_patch(rect2)
ax.text(0.50, 0.37, "Tier 2: Input-Level Attribution (Bypassing Recurrent Buffer)", ha='center', va='center', fontsize=10, weight='bold', color="#1B5E20")
ax.text(0.50, 0.23, "• Gradient × Input: ∑_d |E_{j,d} · ∇_E z_{q,ŷ}|  →  Direct lexical feature sensitivity (Gap: +0.2350)\n"
                    "• Integrated Gradients: ∑_d E_{j,d} · ∫ ∇_E z(α E) dα  →  Axiomatic completeness <0.82% (Gap: +0.2449)\n"
                    "• Decisive Finding: Concentrates 78% mass locally (Dist 0-2); superior deletion faithfulness (p < 0.0001)",
        ha='center', va='center', fontsize=8.3, color="#263238")

ax.set_title("Fig. 6  Two-Tier Explainability Hierarchy: Attention Layer vs. Input Gradient Attribution", fontsize=11, weight='bold', pad=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_explainability_framework.png")
fig.savefig(FIG_DIR / "fig6_explainability_framework.pdf")
plt.close(fig)


print("Generating Figure 7: Faithfulness Evaluation Pipeline...")
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.axis('off')

boxes_f = [
    ("Original Sentence (x)\nTarget Entity Index q\nBaseline Conf C_0", (0.05, 0.40), "#E8F0FE", "#1A73E8"),
    ("Context Masking\nExclude Target (i ≠ q)\nValid Pool: N - 1 tokens", (0.28, 0.40), "#FEF7E0", "#B06000"),
    ("Top-k Attribution\nSelect Top-k Tokens\nErase: E_j = 0", (0.52, 0.65), "#FCE8E6", "#C5221F"),
    ("Random Control\nSelect Random-k Tokens\nErase: E_j = 0", (0.52, 0.15), "#E6F4EA", "#137333"),
    ("Forward-Backward CRF\nMarginal Conf Drop\nDrop_top = max(0, C_0 - C_top)", (0.76, 0.65), "#FCE8E6", "#C5221F"),
    ("Forward-Backward CRF\nMarginal Conf Drop\nDrop_rand = max(0, C_0 - C_rand)", (0.76, 0.15), "#E6F4EA", "#137333"),
]

for text, (x, y), fcol, ecol in boxes_f:
    rect = patches.FancyBboxPatch((x, y), 0.20, 0.25, boxstyle="round,pad=0.015", facecolor=fcol, edgecolor=ecol, lw=1.4)
    ax.add_patch(rect)
    ax.text(x + 0.10, y + 0.125, text, ha='center', va='center', fontsize=8, weight='semibold', color='#202124')

ax.annotate("", xy=(0.28, 0.525), xytext=(0.25, 0.525), **arrow_kwargs)
ax.annotate("", xy=(0.52, 0.775), xytext=(0.48, 0.55), **arrow_kwargs)
ax.annotate("", xy=(0.52, 0.275), xytext=(0.48, 0.50), **arrow_kwargs)
ax.annotate("", xy=(0.76, 0.775), xytext=(0.72, 0.775), **arrow_kwargs)
ax.annotate("", xy=(0.76, 0.275), xytext=(0.72, 0.275), **arrow_kwargs)

# Metric Summary Box
rect_m = patches.FancyBboxPatch((0.25, 0.01), 0.50, 0.10, boxstyle="round,pad=0.01", facecolor="#F3E8FD", edgecolor="#8430CE", lw=1.5)
ax.add_patch(rect_m)
ax.text(0.50, 0.06, "Faithfulness Gap = Drop_top - Drop_rand   |   Win: Gap > 0", ha='center', va='center', fontsize=9, weight='bold', color="#8430CE")

ax.set_title("Fig. 7  Controlled Symmetric Context-Erasure Protocol for Faithfulness Evaluation", fontsize=11, weight='bold', pad=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig7_faithfulness_protocol.png")
fig.savefig(FIG_DIR / "fig7_faithfulness_protocol.pdf")
plt.close(fig)


print("Generating Figure 8: NER Performance Bar Chart...")
fig, ax = plt.subplots(figsize=(7.5, 4.2))

langs = ["Konkani (gom)", "Assamese (as)", "Sinhala (si)", "Macro Average"]
precision = [99.15, 95.83, 96.32, 97.10]
recall = [93.55, 100.00, 99.24, 97.60]
f1 = [96.27, 97.87, 97.76, 97.30]

x = np.arange(len(langs))
width = 0.26

rects1 = ax.bar(x - width, precision, width, label='Precision', color='#1A73E8', edgecolor='#1557B0')
rects2 = ax.bar(x, recall, width, label='Recall', color='#34A853', edgecolor='#1E7E34')
rects3 = ax.bar(x + width, f1, width, label='Entity F1 (Strict)', color='#EA4335', edgecolor='#B31412')

ax.axhline(90.0, color='#EA4335', linestyle='--', linewidth=1.2, label='Target Threshold (90.0%)')
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(langs, fontsize=10, weight='semibold')
ax.set_ylim(85, 103)
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95)
ax.grid(axis='y', linestyle='--', alpha=0.6)

for rect in rects3:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=8.5, weight='bold')

ax.set_title("Fig. 8  Locked Gold-Test Performance Across Languages (Phase 14 Single-Run)", fontsize=11, weight='bold', pad=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig8_ner_performance.png")
fig.savefig(FIG_DIR / "fig8_ner_performance.pdf")
plt.close(fig)


print("Generating Figure 9: Ablation Study Horizontal Waterfall...")
fig, ax = plt.subplots(figsize=(8, 4.4))

conditions = [
    "(g) Subword BPE vs Word+Char",
    "(c) No CRF (Softmax CE)",
    "(b) No Self-Attention",
    "(a) No Char-CNN Encoder",
    "(d) No Multilingual Joint",
    "(e) No Self-Training",
    "(f) No Data Augmentation",
]
deltas = [-7.55, -7.16, -6.62, -5.89, -1.86, -0.37, -0.37]
colors = ['#B71C1C', '#C62828', '#D32F2F', '#E53935', '#F57C00', '#FFB300', '#FFB300']

y_pos = np.arange(len(conditions))
bars = ax.barh(y_pos, deltas, align='center', color=colors, edgecolor='#333333', height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(conditions, fontsize=9.5, weight='semibold')
ax.invert_yaxis()
ax.set_xlabel('Macro Dev Entity F1 Degradation (Δ F1 %)', fontsize=11)
ax.set_xlim(-8.5, 0.5)
ax.grid(axis='x', linestyle='--', alpha=0.6)

for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{width:.2f}%',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(-32, 0), textcoords="offset points",
                ha='left', va='center', fontsize=9, weight='bold', color='white')

ax.set_title("Fig. 9  Component Ablation Impact on Macro Dev F1 (Matched Training Parity)", fontsize=11, weight='bold', pad=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig9_ablation_results.png")
fig.savefig(FIG_DIR / "fig9_ablation_results.pdf")
plt.close(fig)


print("Generating Figure 10: Faithfulness Comparison...")
fig, ax = plt.subplots(figsize=(8, 4.4))

methods = [
    "Integrated Gradients",
    "Grad × Input",
    "Grad-Only Input",
    "Grad × Attn (ReLU)",
    "Grad-Only Attn",
    "Uniform Control",
    "Random Control",
    "Raw Attention",
]
k1_vals = [0.2449, 0.2350, 0.1365, 0.0574, 0.1067, 0.0200, -0.0049, -0.0279]
k3_vals = [0.3396, 0.3057, 0.2105, 0.1128, 0.1680, 0.0350, -0.0080, -0.0617]

x = np.arange(len(methods))
width = 0.38

r1 = ax.bar(x - width/2, k1_vals, width, label='Faithfulness Gap (k=1)', color='#1A73E8', edgecolor='#1557B0')
r2 = ax.bar(x + width/2, k3_vals, width, label='Faithfulness Gap (k=3)', color='#34A853', edgecolor='#1E7E34')

ax.axhline(0.0, color='#333333', linewidth=1.0)
ax.set_ylabel('Macro Faithfulness Gap (Δ)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=9, weight='semibold')
ax.set_ylim(-0.09, 0.38)
ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95)
ax.grid(axis='y', linestyle='--', alpha=0.6)

for r in r1:
    h = r.get_height()
    va = 'bottom' if h >= 0 else 'top'
    y_off = 3 if h >= 0 else -9
    ax.annotate(f'{h:+.3f}',
                xy=(r.get_x() + r.get_width()/2, h),
                xytext=(0, y_off), textcoords="offset points",
                ha='center', va=va, fontsize=7.5, weight='bold')

ax.set_title("Fig. 10  Macro Deletion Faithfulness Gaps Across Attribution Methods and Controls", fontsize=11, weight='bold', pad=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig10_faithfulness_comparison.png")
fig.savefig(FIG_DIR / "fig10_faithfulness_comparison.pdf")
plt.close(fig)


print("Generating Figure 11: Attribution Distance Analysis...")
fig, ax = plt.subplots(figsize=(7.5, 4.2))

bins = ["Dist 0\n(Entity Self)", "Dist 1\n(Adjacent)", "Dist 2", "Dist 3", "Dist 4-5", "Dist > 5\n(Distant)"]
raw_attn = [0.14, 2.10, 6.12, 8.78, 21.34, 61.53]
input_attr = [46.83, 19.69, 11.28, 5.60, 6.09, 10.51]

x = np.arange(len(bins))
width = 0.36

rects1 = ax.bar(x - width/2, raw_attn, width, label='Raw Attention Mass (%)', color='#EA4335', edgecolor='#B31412')
rects2 = ax.bar(x + width/2, input_attr, width, label='Input Attribution Mass (%)', color='#1A73E8', edgecolor='#1557B0')

ax.set_ylabel('Attribution Mass Share (%)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(bins, fontsize=9.5, weight='semibold')
ax.set_ylim(0, 70)
ax.legend(loc='upper center', frameon=True, facecolor='white', framealpha=0.95)
ax.grid(axis='y', linestyle='--', alpha=0.6)

for r in rects1:
    h = r.get_height()
    ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=7.5)

for r in rects2:
    h = r.get_height()
    ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=7.5, weight='bold')

ax.set_title("Fig. 11  Spatial Distribution of Attribution Mass by Relative Token Distance", fontsize=11, weight='bold', pad=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig11_distance_distribution.png")
fig.savefig(FIG_DIR / "fig11_distance_distribution.pdf")
plt.close(fig)

print("All 11 figures and table datasets successfully generated!")
