"""
Print zero_crf protocol tables from faithfulness experiment master results.
"""

from __future__ import annotations

import json
from pathlib import Path

master_path = Path("experiments/faithfulness_improvement/results/faithfulness_experiment_master.json")
with open(master_path, "r", encoding="utf-8") as fp:
    data = json.load(fp)

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

proto = "zero_crf"
print("=" * 115)
print("  PRIMARY PROTOCOL (Zero-Embedding Erasure + Exact CRF Marginal Confidence)")
print("=" * 115)
print(f"{'Language':<10} | {'Method':<22} | {'Top-1':<7} | {'Rand-1':<7} | {'Gap k=1':<8} | {'Gap k=2':<8} | {'Gap k=3':<8} | {'Win k=1':<8} | {'95% CI (k=1)':<18}")
print("-" * 115)

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
    print("-" * 115)

print("\n--- MACRO AVERAGE SUMMARY (zero_crf) ---")
print(f"{'Method':<22} | {'Macro Gap k=1':<14} | {'Macro Win k=1':<14} | {'Macro Gap k=2':<14} | {'Macro Gap k=3':<14}")
print("-" * 85)
for m in methods:
    z = data["macro_summary"][m][proto]
    print(
        f"{m:<22} | "
        f"{z['k=1']['macro_mean_gap']:<+14.4f} | {z['k=1']['macro_win_rate']:<13.1f}% | "
        f"{z['k=2']['macro_mean_gap']:<+14.4f} | {z['k=3']['macro_mean_gap']:<+14.4f}"
    )
