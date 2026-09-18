"""
Programmatic audit of target entity token attribution across all gold-dev entities.

Quantifies:
1. Sample size N of entity predictions evaluated per language.
2. Percentage of instances where the target entity token receives near-zero attribution (<= 0.001, <= 0.01).
3. Mean attribution score assigned to the target entity token itself vs context tokens.
4. Rank distribution of the target entity token (how often is it ranked #1, in top-3, or in the bottom half).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from explainability.attention_extractor import AttentionExtractor
from explainability.erasure_faithfulness_test import load_dev_sentences_with_entities
from explainability.gradient_attention_attribution import GradientAttentionAttributor
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import load_or_create_multilingual_vocabs


def run_target_token_audit() -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    proc_dir = Path("data/processed")
    gold_dir = Path("data/gold")
    checkpoints_dir = Path("checkpoints")
    languages = ("konkani", "assamese", "sinhala")

    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir, languages=languages)

    audit_results: dict[str, Any] = {}

    for lang in languages:
        ckpt_path = checkpoints_dir / f"final_{lang}.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        model = NERMultilingualModel(
            word_vocab_size=len(word_to_id),
            char_vocab_size=len(char_to_id),
            num_tags=9,
            num_languages=3,
            word_embedding_dim=100,
            char_embedding_dim=50,
            lang_embedding_dim=16,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_attention_blocks=1,
            crf_config="shared",
            use_char_cnn=True,
            use_attention=True,
            use_crf=True,
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        lang_id = {"konkani": 0, "assamese": 1, "sinhala": 2}[lang]
        extractor = AttentionExtractor(model, word_to_id, char_to_id, lang, lang_id=lang_id, device=device)
        attributor = GradientAttentionAttributor(extractor)

        dev_sentences = load_dev_sentences_with_entities(gold_dir / lang / "dev.conll")

        # Collect ALL entity instances in gold dev
        all_instances = []
        for sent_data in dev_sentences:
            tokens = sent_data["tokens"]
            for ent in sent_data["entities"]:
                all_instances.append({
                    "tokens": tokens,
                    "token_idx": ent["token_idx"],
                    "gold_tag": ent["tag"],
                    "token": ent["token"],
                })

        N = len(all_instances)
        print(f"\n=======================================================")
        print(f"  Auditing {lang.upper()}: Total Gold-Dev Entity Instances N = {N}")
        print(f"=======================================================")

        grad_attrs = []
        raw_attrs = []
        grad_ranks = []
        raw_ranks = []

        grad_near_zero_001 = 0
        grad_near_zero_01 = 0
        raw_near_zero_001 = 0
        raw_near_zero_01 = 0

        for item in all_instances:
            tokens = item["tokens"]
            q_idx = item["token_idx"]
            seq_len = len(tokens)

            attr_result = attributor.attribute_token(tokens, target_token_idx=q_idx, normalize=True)
            grad_scores = attr_result["attribution_scores"]
            raw_scores = attr_result["raw_attention"]

            target_grad = float(grad_scores[q_idx].item())
            target_raw = float(raw_scores[q_idx].item())
            grad_attrs.append(target_grad)
            raw_attrs.append(target_raw)

            if target_grad <= 0.001:
                grad_near_zero_001 += 1
            if target_grad <= 0.01:
                grad_near_zero_01 += 1

            if target_raw <= 0.001:
                raw_near_zero_001 += 1
            if target_raw <= 0.01:
                raw_near_zero_01 += 1

            grad_ranked = torch.argsort(grad_scores, descending=True).tolist()
            raw_ranked = torch.argsort(raw_scores, descending=True).tolist()

            grad_ranks.append(grad_ranked.index(q_idx) + 1)
            raw_ranks.append(raw_ranked.index(q_idx) + 1)

        grad_top1_count = sum(1 for r in grad_ranks if r == 1)
        grad_top3_count = sum(1 for r in grad_ranks if r <= 3)
        raw_top1_count = sum(1 for r in raw_ranks if r == 1)
        raw_top3_count = sum(1 for r in raw_ranks if r <= 3)

        lang_stats = {
            "N": N,
            "grad_mean_attr": round(sum(grad_attrs) / N, 6),
            "grad_median_attr": round(sorted(grad_attrs)[N // 2], 6),
            "grad_pct_le_001": round(grad_near_zero_001 / N * 100, 2),
            "grad_pct_le_01": round(grad_near_zero_01 / N * 100, 2),
            "grad_pct_top1": round(grad_top1_count / N * 100, 2),
            "grad_pct_top3": round(grad_top3_count / N * 100, 2),
            "grad_mean_rank": round(sum(grad_ranks) / N, 2),
            "raw_mean_attr": round(sum(raw_attrs) / N, 6),
            "raw_pct_le_001": round(raw_near_zero_001 / N * 100, 2),
            "raw_pct_le_01": round(raw_near_zero_01 / N * 100, 2),
            "raw_pct_top1": round(raw_top1_count / N * 100, 2),
            "raw_pct_top3": round(raw_top3_count / N * 100, 2),
            "raw_mean_rank": round(sum(raw_ranks) / N, 2),
        }
        audit_results[lang] = lang_stats

        print(f"[{lang.upper()}] Grad x Attn:")
        print(f"  Mean attribution to target entity token: {lang_stats['grad_mean_attr']:.6f}")
        print(f"  Target token attribution <= 0.001: {lang_stats['grad_pct_le_001']}% ({grad_near_zero_001}/{N})")
        print(f"  Target token attribution <= 0.01:  {lang_stats['grad_pct_le_01']}% ({grad_near_zero_01}/{N})")
        print(f"  Target token is ranked #1: {lang_stats['grad_pct_top1']}% ({grad_top1_count}/{N})")
        print(f"  Target token in top-3:     {lang_stats['grad_pct_top3']}% ({grad_top3_count}/{N})")
        print(f"  Mean rank of target token: {lang_stats['grad_mean_rank']} out of seq_len")

        print(f"[{lang.upper()}] Raw Attention:")
        print(f"  Mean attention to target entity token: {lang_stats['raw_mean_attr']:.6f}")
        print(f"  Target token attention <= 0.001: {lang_stats['raw_pct_le_001']}% ({raw_near_zero_001}/{N})")
        print(f"  Target token attention <= 0.01:  {lang_stats['raw_pct_le_01']}% ({raw_near_zero_01}/{N})")
        print(f"  Target token is ranked #1: {lang_stats['raw_pct_top1']}% ({raw_top1_count}/{N})")
        print(f"  Target token in top-3:     {lang_stats['raw_pct_top3']}% ({raw_top3_count}/{N})")

    out_json = Path("reports/experiment_logs/target_token_attribution_audit.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(audit_results, fp, indent=2)

    return audit_results


if __name__ == "__main__":
    run_target_token_audit()
