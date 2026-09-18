"""
Print comprehensive tables from faithfulness experiment master results.
"""

from __future__ import annotations

import json
from pathlib import Path

master_path = Path("experiments/faithfulness_improvement/results/faithfulness_experiment_master.json")
with open(master_path, "r", encoding="utf-8") as fp:
    data = json.load(fp)

print("=" * 110)
print("  FAITHFULNESS IMPROVEMENT EXPERIMENT: MASTER SUMMARY TABLE")
print("=" * 110)

languages = ["konkani", "assamese", "sinhala"]
methods = [
    "raw_attention",
    "grad_x_attn_abs",
    "grad_x_attn_relu",
    "grad_x_attn_lsm",
    "grad_x_attn_crf",
    "grad_x_input",
    "integrated_gradients",
    "raw_head1",
]

for proto, proto_title in [
    ("zero_crf", "PRIMARY PROTOCOL (Zero-Embedding Erasure + Exact CRF Marginal Confidence)"),
    ("unk_emission", "BASELINE PROTOCOL (UNK-Token Erasure + Emission Softmax Confidence)"),
]:
    print(f"\n--- {proto_title} ---")
    print(f"{'Language':<10} | {'Method':<22} | {'Top-1':<7} | {'Rand-1':<7} | {'Gap k=1':<8} | {'Gap k=2':<8} | {'Gap k=3':<8} | {'Win k=1':<8} | {'95% CI (k=1)':<18}")
    print("-" * 110)

    for lang in languages:
        for m in methods:
            m_data = data["languages"][lang][m][proto]
            k1 = m_data["k=1"]
            k2 = m_data["k=2"]
            k3 = m_data["k=3"]
            ci_str = f"[{k1['ci_lower_95']:+.4f}, {k1['ci_upper_95']:+.4f}]"
            print(
                f"{lang.capitalize():<10} | {m:<22} | "
                f"{k1['mean_top_drop']:<7.4f} | {k1['mean_rand_drop']:<7.4f} | "
                f"{k1['mean_gap']:<+8.4f} | {k2['mean_gap']:<+8.4f} | {k3['mean_gap']:<+8.4f} | "
                f"{k1['win_rate']:<7.1f}% | {ci_str:<18}"
            )
        print("-" * 110)

    print("\n--- MACRO AVERAGE SUMMARY ---")
    print(f"{'Protocol':<15} | {'Method':<22} | {'Macro Gap k=1':<14} | {'Macro Win k=1':<14} | {'Macro Gap k=2':<14} | {'Macro Gap k=3':<14}")
    print("-" * 105)
    for m in methods:
        z = data["macro_summary"][m][proto]
        print(
            f"{proto:<15} | {m:<22} | "
            f"{z['k=1']['macro_mean_gap']:<+14.4f} | {z['k=1']['macro_win_rate']:<13.1f}% | "
            f"{z['k=2']['macro_mean_gap']:<+14.4f} | {z['k=3']['macro_mean_gap']:<+14.4f}"
        )

print("\n" + "=" * 110)
print("  CONTEXT DISTANCE PROFILE (Average Attribution Mass % by Distance from Query)")
print("=" * 110)
print(f"{'Language':<10} | {'Dist = 1 (Immediate)':<22} | {'Dist = 2':<15} | {'Dist 3-5':<15} | {'Dist > 5 (Distant)':<20}")
print("-" * 90)
for lang in languages:
    dp = data["distance_analysis"][lang]
    print(f"{lang.capitalize():<10} | {dp['dist_1']:<22.2f}% | {dp['dist_2']:<15.2f}% | {dp['dist_3_5']:<15.2f}% | {dp['dist_gt_5']:<20.2f}%")

print("\n" + "=" * 110)
print("  4-HEAD ATTENTION QUANTITATIVE PROFILES")
print("=" * 110)
print(f"{'Language':<10} | {'Head':<6} | {'Entropy':<8} | {'Avg Dist':<9} | {'Self %':<8} | {'Immediate %':<12} | {'Distant %':<10}")
print("-" * 75)
for lang in languages:
    for h in range(4):
        hm = data["head_analysis"][lang][str(h)]
        print(f"{lang.capitalize():<10} | Head {h:<1} | {hm['entropy']:<8.2f} | {hm['avg_distance']:<9.2f} | {hm['self_prop']:<8.2f}% | {hm['immediate_prop']:<12.2f}% | {hm['distant_prop']:<10.2f}%")
    print("-" * 75)

print("\n" + "=" * 110)
print("  ENTITY-TYPE BREAKDOWN (Mean Faithfulness Gap k=1 under zero_crf)")
print("=" * 110)
print(f"{'Language':<10} | {'Entity':<8} | {'Raw Attn':<10} | {'GradxAttn Abs':<14} | {'GradxAttn ReLU':<15} | {'GradxInput':<12} | {'Head 1':<10}")
print("-" * 90)
for lang in languages:
    for ent, m_dict in data["entity_type_analysis"][lang].items():
        print(
            f"{lang.capitalize():<10} | {ent:<8} | "
            f"{m_dict.get('raw_attention', 0.0):<+10.4f} | "
            f"{m_dict.get('grad_x_attn_abs', 0.0):<+14.4f} | "
            f"{m_dict.get('grad_x_attn_relu', 0.0):<+15.4f} | "
            f"{m_dict.get('grad_x_input', 0.0):<+12.4f} | "
            f"{m_dict.get('raw_head1', 0.0):<+10.4f}"
        )
