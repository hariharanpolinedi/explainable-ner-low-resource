import json, os, numpy as np, matplotlib.pyplot as plt

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PHASE14 = os.path.join(BASE_DIR, 'reports', 'experiment_logs', 'phase14_test_eval_lock.json')
FAITHFULL = os.path.join(BASE_DIR, 'reports', 'experiment_logs', 'phase15_faithfulness_four_language_raw.json')
TARGET_AUDIT = os.path.join(BASE_DIR, 'reports', 'experiment_logs', 'target_token_attribution_audit.json')
FIG_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def save(fig_path):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

# 1. Entity‑level F1 (include Kokborok manually)
with open(PHASE14, 'r', encoding='utf-8') as f:
    data = json.load(f)
langs_lock = ['konkani', 'assamese', 'sinhala']
entity_f1 = [data['languages'][l]['final_model']['overall']['entity_f1']*100 for l in langs_lock]
entity_f1.append(82.35)  # Kokborok official F1
langs = langs_lock + ['kokborok']
plt.figure(figsize=(6,4))
plt.bar(langs, entity_f1, color='#4B8BBE')
plt.ylabel('Entity-level F1 (%)')
plt.title('Entity-level F1 by Language')
plt.ylim(0,100)
save(os.path.join(FIG_DIR, 'ner_entity_f1_by_language.png'))

# 2. Precision / Recall / F1 (include Kokborok manually)
precisions = []
recalls = []
f1s = []
for l in langs_lock:
    overall = data['languages'][l]['final_model']['overall']
    precisions.append(overall['entity_precision']*100)
    recalls.append(overall['entity_recall']*100)
    f1s.append(overall['entity_f1']*100)
precisions.append(77.78)
recalls.append(87.50)
f1s.append(82.35)
import numpy as np
x = np.arange(len(langs))
width = 0.25
plt.figure(figsize=(8,4))
plt.bar(x-width, precisions, width, label='Precision', color='#FF6F61')
plt.bar(x, recalls, width, label='Recall', color='#6B5B95')
plt.bar(x+width, f1s, width, label='F1', color='#88B04B')
plt.xticks(x, langs)
plt.ylabel('Score (%)')
plt.title('Precision, Recall, and F1 by Language')
plt.ylim(0,100)
plt.legend()
save(os.path.join(FIG_DIR, 'ner_precision_recall_f1_by_language.png'))

# 3. Final vs Joint Baseline (exclude Kokborok)
joint_f1 = []
final_f1 = []
joint_langs = []
for l in ['konkani','assamese','sinhala']:
    joint_f1.append(data['languages'][l]['joint_baseline']['overall']['entity_f1']*100)
    final_f1.append(data['languages'][l]['final_model']['overall']['entity_f1']*100)
    joint_langs.append(l)
indices = np.arange(len(joint_langs))
plt.figure(figsize=(6,4))
plt.bar(indices-width/2, joint_f1, width, label='Joint Baseline', color='#FFA500')
plt.bar(indices+width/2, final_f1, width, label='Final', color='#008080')
plt.xticks(indices, joint_langs)
plt.ylabel('Entity-level F1 (%)')
plt.title('Final vs Joint Baseline F1')
plt.ylim(0,100)
plt.legend()
save(os.path.join(FIG_DIR, 'final_vs_joint_baseline_f1.png'))

# 4. Faithfulness attribution comparison (faithfulness_gap) – combined grid
with open(FAITHFULL, 'r', encoding='utf-8') as f:
    faith = json.load(f)['faithfulness']
methods = ['grad_x_attn','raw_attention']
ks = ['k=1','k=2','k=3']
plt.figure(figsize=(12,8))
for i, (lang, vals) in enumerate(faith.items()):
    ax = plt.subplot(2,2,i+1)
    for method in methods:
        if method in vals:
            gaps = [vals[method][k]['faithfulness_gap'] for k in ks]
            ax.plot([1,2,3], gaps, marker='o', label=method.replace('_',' ').title())
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(ks)
    ax.set_ylabel('Faithfulness Gap')
    ax.set_title(lang.title())
    ax.legend(fontsize='small')
plt.suptitle('Faithfulness Gap by Method and k')
save(os.path.join(FIG_DIR, 'faithfulness_attribution_comparison.png'))

# 5. Target token near‑zero attribution percentages
with open(TARGET_AUDIT, 'r', encoding='utf-8') as f:
    audit = json.load(f)
langs_a = list(audit.keys())
grad_near = [audit[l]['grad_pct_le_001'] for l in langs_a]
raw_near = [audit[l]['raw_pct_le_001'] for l in langs_a]
ind = np.arange(len(langs_a))
width = 0.35
plt.figure(figsize=(6,4))
plt.bar(ind, grad_near, width, label='Grad × Attention', color='#F7CAC9')
plt.bar(ind+width, raw_near, width, label='Raw Attention', color='#92A8D1')
plt.xticks(ind+width/2, langs_a)
plt.ylabel('% ≤ 0.001')
plt.title('Near‑zero Attribution Percentage')
plt.ylim(0,100)
plt.legend()
save(os.path.join(FIG_DIR, 'target_token_near_zero_attribution.png'))

# 6. Target token rank distribution (mean rank)
grad_rank = [audit[l]['grad_mean_rank'] for l in langs_a]
raw_rank = [audit[l]['raw_mean_rank'] for l in langs_a]
plt.figure(figsize=(6,4))
plt.bar(ind, grad_rank, width, label='Grad × Attention', color='#F7CAC9')
plt.bar(ind+width, raw_rank, width, label='Raw Attention', color='#92A8D1')
plt.xticks(ind+width/2, langs_a)
plt.ylabel('Mean Rank')
plt.title('Target Token Rank Distribution')
plt.legend()
save(os.path.join(FIG_DIR, 'target_token_rank_distribution.png'))

print('All required figures generated.')
