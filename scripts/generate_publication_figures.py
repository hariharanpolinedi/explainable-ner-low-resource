"""
Generate high-resolution publication-quality figures for Sections 7 and 8.
All numerical data is strictly derived from authoritative project artifacts:
- reports/experiment_logs/phase14_test_eval_lock.json
- reports/experiment_logs/phase15_faithfulness_four_language_raw.json
- reports/experiment_logs/target_token_attribution_audit.json
- reports/kokborok_data_sufficiency_check.md
- reports/phase14_confusion_matrices/ and reports/kokborok_confusion_matrix.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image

# Set global matplotlib styles
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
})

FIG_DIRS = ['figures', os.path.join('reports', 'figures'), os.path.join('paper', 'figures')]
for d in FIG_DIRS:
    os.makedirs(d, exist_ok=True)

def save_fig(name):
    for d in FIG_DIRS:
        png_path = os.path.join(d, f"{name}.png")
        pdf_path = os.path.join(d, f"{name}.pdf")
        plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white', edgecolor='none', transparent=False)
        plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.1, facecolor='white', edgecolor='none', transparent=False)
        im = Image.open(png_path)
        if im.mode != 'RGB':
            im.convert('RGB').save(png_path, 'PNG', optimize=True)
    plt.close()
    print(f"Saved {name}.png and {name}.pdf")

# -------------------------------------------------------------------------
# FIGURE A: Entity F1 by Language
# -------------------------------------------------------------------------
def make_figure_a():
    langs = ['Konkani', 'Assamese', 'Sinhala', 'Kokborok']
    f1_scores = [96.27, 97.87, 97.76, 82.35]
    colors = ['#1d3557', '#2a9d8f', '#e76f51', '#6b2d5c']

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(langs, f1_scores, color=colors, width=0.55, edgecolor='#222222', linewidth=0.8, zorder=3)
    ax.set_ylabel('Entity-Level F1 Score (%)')
    ax.set_xlabel('Target Language')
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis='y', zorder=0)

    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2.0, f"{val:.2f}%",
                ha='center', va='bottom', fontweight='bold', fontsize=10.5, color='#111111')

    plt.title('Official Gold-Test Entity-Level F1 Performance by Language', pad=12, fontweight='bold')
    plt.tight_layout()
    save_fig('fig_a_entity_f1')

# -------------------------------------------------------------------------
# FIGURE B: Precision, Recall, and F1 by Language
# -------------------------------------------------------------------------
def make_figure_b():
    langs = ['Konkani', 'Assamese', 'Sinhala', 'Kokborok']
    precision = [99.15, 95.83, 96.32, 77.78]
    recall = [93.55, 100.00, 99.24, 87.50]
    f1 = [96.27, 97.87, 97.76, 82.35]

    x = np.arange(len(langs))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    rects1 = ax.bar(x - width, precision, width, label='Precision', color='#3a86ff', edgecolor='#222222', linewidth=0.7, zorder=3)
    rects2 = ax.bar(x, recall, width, label='Recall', color='#ffbe0b', edgecolor='#222222', linewidth=0.7, zorder=3)
    rects3 = ax.bar(x + width, f1, width, label='Entity F1', color='#38b000', edgecolor='#222222', linewidth=0.7, zorder=3)

    ax.set_ylabel('Score (%)')
    ax.set_xlabel('Target Language')
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontweight='medium')
    ax.set_ylim(0, 115)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc', loc='lower right')
    ax.grid(axis='y', zorder=0)

    for rects in [rects1, rects2, rects3]:
        for r in rects:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + 1.5, f"{h:.1f}",
                    ha='center', va='bottom', fontsize=8.5, rotation=0)

    plt.title('Precision, Recall, and Entity F1 across Locked Gold-Test Splits', pad=12, fontweight='bold')
    plt.tight_layout()
    save_fig('fig_b_precision_recall_f1')

# -------------------------------------------------------------------------
# FIGURE C: Final Model vs Joint Baseline
# -------------------------------------------------------------------------
def make_figure_c():
    langs = ['Konkani', 'Assamese', 'Sinhala']
    joint_f1 = [95.87, 97.86, 97.04]
    final_f1 = [96.27, 97.87, 97.76]
    deltas = [f1 - j for f1, j in zip(final_f1, joint_f1)]

    x = np.arange(len(langs))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    rects1 = ax.bar(x - width/2, joint_f1, width, label='Joint Baseline', color='#8d99ae', edgecolor='#222222', linewidth=0.7, zorder=3)
    rects2 = ax.bar(x + width/2, final_f1, width, label='Final Model', color='#2b2d42', edgecolor='#222222', linewidth=0.7, zorder=3)

    ax.set_ylabel('Entity F1 (%)')
    ax.set_xlabel('Target Language')
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontweight='medium')
    ax.set_ylim(0, 115)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc', loc='lower right')
    ax.grid(axis='y', zorder=0)

    for i, (r1, r2, d) in enumerate(zip(rects1, rects2, deltas)):
        ax.text(r1.get_x() + r1.get_width()/2, r1.get_height() + 1.8, f"{r1.get_height():.2f}%", ha='center', va='bottom', fontsize=8.5)
        ax.text(r2.get_x() + r2.get_width()/2, r2.get_height() + 1.8, f"{r2.get_height():.2f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax.annotate(f"$\\Delta$ = +{d:.2f} pp", xy=(x[i], max(r1.get_height(), r2.get_height()) + 7.5),
                    ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#1d3557')

    plt.title('Final Model vs. Multilingual Joint Baseline (Verified Cohort)', pad=12, fontweight='bold')
    plt.tight_layout()
    save_fig('fig_c_final_vs_joint_baseline')

# -------------------------------------------------------------------------
# FIGURE D: Actual Attribution Heatmap (Gold-Dev Output)
# -------------------------------------------------------------------------
def make_figure_d():
    tokens = ['Lunglei', 'song', 'advent', 'Christmas\n[Target: B-MISC]', 'khlaina', 'thangwi', 'tongo']
    # Actual model values on this Gold-Dev instance:
    raw_attn = [0.1463, 0.2035, 0.1423, 0.0060, 0.1394, 0.1567, 0.2058]
    grad_attn = [0.2030, 0.1818, 0.1861, 0.0337, 0.1376, 0.1411, 0.1168]
    matrix = np.array([raw_attn, grad_attn])

    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    cax = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0.0, vmax=0.22)

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=10.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Raw Attention', 'Gradient × Attention'], fontsize=11, fontweight='medium')

    # Add numeric annotations inside cells
    for i in range(2):
        for j in range(len(tokens)):
            val = matrix[i, j]
            text_color = '#ffffff' if val > 0.15 else '#000000'
            weight = 'bold' if j == 3 else 'normal'
            ax.text(j, i, f"{val:.4f}", ha='center', va='center', color=text_color, fontsize=10, fontweight=weight)

    cbar = fig.colorbar(cax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('Attribution Weight / Magnitude', fontsize=10.5)

    plt.title('Attribution Distribution on Gold-Dev Instance (Target: Christmas, B-MISC)', pad=12, fontweight='bold')
    plt.tight_layout()
    save_fig('fig_d_attribution_heatmap')

# -------------------------------------------------------------------------
# FIGURE E: Target-Token Near-Zero Attribution Distribution
# -------------------------------------------------------------------------
def make_figure_e():
    langs = ['Konkani\n(N=125)', 'Assamese\n(N=139)', 'Sinhala\n(N=131)']
    grad_near = [92.80, 98.56, 98.47]
    raw_near = [93.60, 98.56, 99.24]

    x = np.arange(len(langs))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    rects1 = ax.bar(x - width/2, grad_near, width, label='Gradient × Attention', color='#0077b6', edgecolor='#222222', linewidth=0.7, zorder=3)
    rects2 = ax.bar(x + width/2, raw_near, width, label='Raw Attention', color='#f77f00', edgecolor='#222222', linewidth=0.7, zorder=3)

    ax.set_ylabel(r'Target Tokens with Attr. $\leq 0.001$ (%)', labelpad=8)
    ax.set_xlabel('Target Language Cohort (Gold-Dev Entity Audit)', labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontweight='medium')
    ax.set_ylim(0, 115)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc', loc='lower right')
    ax.grid(axis='y', zorder=0)

    for rects in [rects1, rects2]:
        for r in rects:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + 2.0, f"{h:.2f}%",
                    ha='center', va='bottom', fontsize=9.0, fontweight='bold')

    plt.title(r'Target-Token Attribution Sparsity ($\leq 0.001$)', pad=14, fontweight='bold')
    plt.tight_layout()
    save_fig('fig_e_target_token_near_zero')

# -------------------------------------------------------------------------
# FIGURE F: Target-Token Mean Rank Distribution
# -------------------------------------------------------------------------
def make_figure_f():
    langs = ['Konkani\n(N=125)', 'Assamese\n(N=139)', 'Sinhala\n(N=131)']
    grad_rank = [16.94, 17.17, 20.57]
    raw_rank = [17.26, 17.30, 20.63]

    x = np.arange(len(langs))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    rects1 = ax.bar(x - width/2, grad_rank, width, label='Gradient × Attention', color='#457b9d', edgecolor='#222222', linewidth=0.7, zorder=3)
    rects2 = ax.bar(x + width/2, raw_rank, width, label='Raw Attention', color='#e63946', edgecolor='#222222', linewidth=0.7, zorder=3)

    ax.set_ylabel('Mean Rank of Target Entity Token', labelpad=8)
    ax.set_xlabel('Target Language Cohort (Gold-Dev Entity Audit)', labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontweight='medium')
    ax.set_ylim(0, 26)
    ax.set_yticks(np.arange(0, 26, 5))
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc', loc='upper left')
    ax.grid(axis='y', zorder=0)

    for rects in [rects1, rects2]:
        for r in rects:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + 0.5, f"Rank {h:.2f}",
                    ha='center', va='bottom', fontsize=9)

    plt.title('Mean Target-Token Rank under Contextual Attribution Ordering', pad=14, fontweight='bold')
    plt.tight_layout()
    save_fig('fig_f_target_token_rank')

# -------------------------------------------------------------------------
# FIGURE G: Faithfulness Line Graph
# -------------------------------------------------------------------------
def make_figure_g():
    # Authoritative Phase-15 values:
    # Konkani:
    #   Raw: k=1: +0.0363, k=2: +0.0412, k=3: +0.0316
    #   Grad: k=1: +0.0190, k=2: +0.0250, k=3: +0.0127
    # Assamese:
    #   Raw: k=1: -0.0135, k=2: -0.0150, k=3: -0.0130
    #   Grad: k=1: -0.0232, k=2: -0.0220, k=3: -0.0143
    # Sinhala:
    #   Raw: k=1: +0.0107, k=2: -0.0011, k=3: +0.0029
    #   Grad: k=1: +0.0058, k=2: -0.0080, k=3: -0.0051
    # Kokborok:
    #   Raw: k=1: -0.0132, k=2: -0.0198, k=3: -0.0381
    #   Grad: k=1: -0.0141, k=2: -0.0365, k=3: -0.0539

    ks = [1, 2, 3]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.8), sharey=True)

    data = {
        'Konkani (N=100)': {
            'Raw': [0.0363, 0.0412, 0.0316],
            'Grad': [0.0190, 0.0250, 0.0127],
        },
        'Assamese (N=100)': {
            'Raw': [-0.0135, -0.0150, -0.0130],
            'Grad': [-0.0232, -0.0220, -0.0143],
        },
        'Sinhala (N=100)': {
            'Raw': [0.0107, -0.0011, 0.0029],
            'Grad': [0.0058, -0.0080, -0.0051],
        },
        'Kokborok (N=17)': {
            'Raw': [-0.0132, -0.0198, -0.0381],
            'Grad': [-0.0141, -0.0365, -0.0539],
        },
    }

    for idx, (lang, methods) in enumerate(data.items()):
        ax = axes[idx]
        ax.axhline(0, color='#888888', linestyle=':', linewidth=1.0, zorder=1)
        ax.plot(ks, methods['Raw'], marker='o', linewidth=2.0, color='#2a9d8f', label='Raw Attention', zorder=3)
        ax.plot(ks, methods['Grad'], marker='s', linewidth=2.0, color='#e76f51', linestyle='--', label='Gradient × Attn', zorder=3)

        ax.set_title(lang, fontsize=11, fontweight='bold')
        ax.set_xlabel('Context Erasure Size ($k$)')
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['$k=1$', '$k=2$', '$k=3$'])
        ax.grid(True, zorder=0)

        if idx == 0:
            ax.set_ylabel(r'Faithfulness Gap ($\Delta_k = \Delta_{\mathrm{top}} - \Delta_{\mathrm{rand}}$)')
            ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=9.5, loc='upper right')

    axes[0].set_ylim(-0.065, 0.055)
    plt.suptitle(r'Empirical Faithfulness Gap Across Context Erasure Levels $k \in \{1, 2, 3\}$ (Phase 15)', y=1.03, fontweight='bold', fontsize=13)
    plt.tight_layout()
    save_fig('fig_g_faithfulness_line')

# -------------------------------------------------------------------------
# FIGURE H: Faithfulness Heatmap
# -------------------------------------------------------------------------
def make_figure_h():
    langs = ['Konkani', 'Assamese', 'Sinhala', 'Kokborok']
    raw_matrix = np.array([
        [0.0363, 0.0412, 0.0316],
        [-0.0135, -0.0150, -0.0130],
        [0.0107, -0.0011, 0.0029],
        [-0.0132, -0.0198, -0.0381]
    ])
    grad_matrix = np.array([
        [0.0190, 0.0250, 0.0127],
        [-0.0232, -0.0220, -0.0143],
        [0.0058, -0.0080, -0.0051],
        [-0.0141, -0.0365, -0.0539]
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)

    cax1 = ax1.imshow(raw_matrix, cmap='coolwarm', vmin=-0.06, vmax=0.06, aspect='auto')
    ax1.set_title('Raw Attention Faithfulness Gap', fontweight='bold', pad=10)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(['$k=1$', '$k=2$', '$k=3$'])
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(langs, fontweight='medium')

    for i in range(4):
        for j in range(3):
            val = raw_matrix[i, j]
            color = '#ffffff' if abs(val) > 0.035 else '#000000'
            sign = '+' if val > 0 else ''
            ax1.text(j, i, f"{sign}{val:.4f}", ha='center', va='center', color=color, fontsize=10, fontweight='bold')

    cax2 = ax2.imshow(grad_matrix, cmap='coolwarm', vmin=-0.06, vmax=0.06, aspect='auto')
    ax2.set_title('Gradient × Attention Faithfulness Gap', fontweight='bold', pad=10)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['$k=1$', '$k=2$', '$k=3$'])

    for i in range(4):
        for j in range(3):
            val = grad_matrix[i, j]
            color = '#ffffff' if abs(val) > 0.035 else '#000000'
            sign = '+' if val > 0 else ''
            ax2.text(j, i, f"{sign}{val:.4f}", ha='center', va='center', color=color, fontsize=10, fontweight='bold')

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.18, 0.02, 0.65])
    cbar = fig.colorbar(cax2, cax=cbar_ax)
    cbar.set_label(r'Faithfulness Gap ($\Delta_k$)', fontsize=10.5)

    plt.suptitle(r'Faithfulness Gap Comparison (Language × Erasure Size $k$)', y=1.02, fontweight='bold', fontsize=12.5)
    save_fig('fig_h_faithfulness_heatmap')

# -------------------------------------------------------------------------
# FIGURE I: Resource & Entity Density Characteristics
# -------------------------------------------------------------------------
def make_figure_i():
    langs = ['Konkani', 'Assamese', 'Sinhala', 'Kokborok']
    train_entities = [590, 690, 519, 30]
    train_tokens = [13710, 14649, 16541, 6711]
    density_per_k = [e / (t / 1000) for e, t in zip(train_entities, train_tokens)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Panel 1: Gold Training Entity Count
    bars1 = ax1.bar(langs, train_entities, color=['#457b9d', '#2a9d8f', '#e76f51', '#e63946'],
                    width=0.55, edgecolor='#222222', linewidth=0.8, zorder=3)
    ax1.set_ylabel('Total Gold Training Entities ($N$)')
    ax1.set_title('Gold Training Entity Annotations', fontweight='bold', pad=10)
    ax1.set_ylim(0, 780)
    ax1.grid(axis='y', zorder=0)

    for b, val in zip(bars1, train_entities):
        ax1.text(b.get_x() + b.get_width() / 2, val + 15, f"{val}",
                 ha='center', va='bottom', fontweight='bold', fontsize=10.5)

    # Panel 2: Entity Density per 1,000 Tokens
    bars2 = ax2.bar(langs, density_per_k, color=['#457b9d', '#2a9d8f', '#e76f51', '#e63946'],
                    width=0.55, edgecolor='#222222', linewidth=0.8, zorder=3)
    ax2.set_ylabel('Entity Density (Entities per 1,000 Tokens)')
    ax2.set_title('Gold Training Entity Supervision Density', fontweight='bold', pad=10)
    ax2.set_ylim(0, 55)
    ax2.grid(axis='y', zorder=0)

    for b, val in zip(bars2, density_per_k):
        ax2.text(b.get_x() + b.get_width() / 2, val + 1.0, f"{val:.1f}",
                 ha='center', va='bottom', fontweight='bold', fontsize=10.5)

    plt.suptitle('Training Resource & Entity Density Audit (Documented Gold Standard)', y=1.02, fontweight='bold', fontsize=12.5)
    plt.tight_layout()
    save_fig('fig_i_data_density')

# -------------------------------------------------------------------------
# FIGURE K: Four-Language Entity Confusion Matrix Grid
# -------------------------------------------------------------------------
def make_figure_k():
    paths = [
        ('Konkani (gom)', 'reports/phase14_confusion_matrices/konkani_entity_matrix.png'),
        ('Assamese (asm)', 'reports/phase14_confusion_matrices/assamese_entity_matrix.png'),
        ('Sinhala (sin)', 'reports/phase14_confusion_matrices/sinhala_entity_matrix.png'),
        ('Kokborok (kok)', 'reports/phase14_confusion_matrices/kokborok_entity_matrix.png'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.8))
    axes = axes.flatten()

    for ax, (title, path) in zip(axes, paths):
        if os.path.exists(path):
            img = Image.open(path)
            ax.imshow(img)
            ax.set_title(title, fontweight='bold', fontsize=12, pad=6)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f"Artifact not found:\n{path}", ha='center', va='center')
            ax.axis('off')

    plt.suptitle('Entity-Level Prediction Confusion Matrices across Locked Gold-Test Splits', y=0.99, fontweight='bold', fontsize=13.0)
    plt.tight_layout()
    save_fig('fig_k_confusion_matrices')

if __name__ == '__main__':
    print("Generating all publication figures...")
    make_figure_a()
    make_figure_b()
    make_figure_c()
    make_figure_d()
    make_figure_e()
    make_figure_f()
    make_figure_g()
    make_figure_h()
    make_figure_i()
    make_figure_k()
    print("All figures successfully generated in figures/ and reports/figures/")
