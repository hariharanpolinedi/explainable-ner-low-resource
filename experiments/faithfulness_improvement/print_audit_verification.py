"""
Print independent audit verification tables.
"""

from __future__ import annotations

import json
from pathlib import Path

with open("experiments/faithfulness_improvement/results/final_faithfulness_audit.json", encoding="utf-8") as f:
    audit = json.load(f)

with open("experiments/faithfulness_improvement/results/faithfulness_experiment_master.json", encoding="utf-8") as f:
    master = json.load(f)

print("=" * 105)
print("  INDEPENDENT RECALCULATION AUDIT (Primary Protocol: Zero-Embedding + CRF Marginals)")
print("=" * 105)
print(f"{'Condition / Metric':<30} | {'Reported':<12} | {'Recalculated':<14} | {'Difference':<12} | {'Status':<10}")
print("-" * 105)

for m in ["integrated_gradients", "grad_x_input", "grad_x_attn_relu", "raw_attention"]:
    print(f"\n--- Method: {m.upper()} ---")
    for k in [1, 2, 3]:
        # Konkani
        rep_k = master["languages"]["konkani"][m]["zero_crf"][f"k={k}"]["mean_gap"]
        rec_k = audit["recalculated_metrics"]["konkani"][m][f"k={k}"]["mean_gap"]
        diff_k = rec_k - rep_k
        print(f"Konkani k={k:<22} | {rep_k:<+12.4f} | {rec_k:<+14.4f} | {diff_k:<+12.4f} | {'MATCH' if abs(diff_k) < 1e-4 else 'DIFF':<10}")

        # Assamese
        rep_a = master["languages"]["assamese"][m]["zero_crf"][f"k={k}"]["mean_gap"]
        rec_a = audit["recalculated_metrics"]["assamese"][m][f"k={k}"]["mean_gap"]
        diff_a = rec_a - rep_a
        print(f"Assamese k={k:<21} | {rep_a:<+12.4f} | {rec_a:<+14.4f} | {diff_a:<+12.4f} | {'MATCH' if abs(diff_a) < 1e-4 else 'DIFF':<10}")

        # Sinhala
        rep_s = master["languages"]["sinhala"][m]["zero_crf"][f"k={k}"]["mean_gap"]
        rec_s = audit["recalculated_metrics"]["sinhala"][m][f"k={k}"]["mean_gap"]
        diff_s = rec_s - rep_s
        print(f"Sinhala k={k:<22} | {rep_s:<+12.4f} | {rec_s:<+14.4f} | {diff_s:<+12.4f} | {'MATCH' if abs(diff_s) < 1e-4 else 'DIFF':<10}")

        # Macro
        rep_m = master["macro_summary"][m]["zero_crf"][f"k={k}"]["macro_mean_gap"]
        rec_m = audit["macro_recalculated"][m][f"k={k}"]["macro_mean_gap"]
        diff_m = rec_m - rep_m
        print(f"Macro k={k:<24} | {rep_m:<+12.4f} | {rec_m:<+14.4f} | {diff_m:<+12.4f} | {'MATCH' if abs(diff_m) < 1e-4 else 'DIFF':<10}")

        # Macro Win Rate
        rep_w = master["macro_summary"][m]["zero_crf"][f"k={k}"]["macro_win_rate"]
        rec_w = audit["macro_recalculated"][m][f"k={k}"]["macro_win_rate"]
        diff_w = rec_w - rep_w
        print(f"Macro Win Rate k={k:<15} | {rep_w:<11.2f}% | {rec_w:<13.2f}% | {diff_w:<+11.2f}% | {'MATCH' if abs(diff_w) < 1e-2 else 'DIFF':<10}")

print("\n" + "=" * 105)
print("  MULTIPLE TESTING CORRECTION AUDIT (P = 10,000 Permutations)")
print("=" * 105)
print(f"{'Hypothesis Test':<40} | {'Exact Raw p':<14} | {'Holm-Bonferroni':<18} | {'Benjamini-Hochberg':<20} | {'Sig (alpha=0.05)':<15}")
print("-" * 115)
for k, v in audit["multiple_testing"]["holm_bonferroni"].items():
    raw_p = v["raw_p"]
    holm_p = v["holm_adj_p"]
    bh_p = audit["multiple_testing"]["benjamini_hochberg"][k]["bh_adj_p"]
    raw_str = f"{raw_p:.6f}" if raw_p > 0 else "< 0.0001 (0/10000)"
    holm_str = f"{holm_p:.6f}" if holm_p > 0 else "< 0.0001"
    bh_str = f"{bh_p:.6f}" if bh_p > 0 else "< 0.0001"
    print(f"{k:<40} | {raw_str:<14} | {holm_str:<18} | {bh_str:<20} | {str(v['sig_05']):<15}")

out_summary_file = Path("experiments/faithfulness_improvement/results/audit_summary.txt")
with open(out_summary_file, "w", encoding="utf-8") as out_fp:
    out_fp.write("\n" + "=" * 105 + "\n")
    out_fp.write("  SAMPLE REAL TARGET LEAKAGE AUDIT CASES (From First Instances)\n")
    out_fp.write("=" * 105 + "\n")
    for lang in ["konkani", "assamese", "sinhala"]:
        out_fp.write(f"\n[{lang.upper()}] Real Sentence Leakage Checks:\n")
        for ex in audit["target_leakage_audit"][lang][:5]:
            out_fp.write(f"  Sentence:       {ex['sentence'][:80]}...\n")
            out_fp.write(f"  Target Entity:  '{ex['target_entity']}' (token_idx = {ex['target_idx']})\n")
            out_fp.write(f"  In Candidates?  {ex['target_in_candidates']} (MUST BE FALSE)\n")
            out_fp.write(f"  Top-1 Context:  '{ex['top1_token']}' (token_idx = {ex['top1_idx']})\n")
            out_fp.write(f"  In Top-1?       {ex['target_in_top1']} (MUST BE FALSE)\n\n")

print(f"\n[SUCCESS] Target leakage cases written to {out_summary_file}")
