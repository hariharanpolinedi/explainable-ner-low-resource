"""
Audit script to verify the erasure test design in explainability/erasure_faithfulness_test.py.

Answers:
1. Exact selection code path for top-k vs random-k.
2. Can target entity token be selected as top-k?
3. Can target entity token be selected as random-k?
4. Quantify the fraction of random-k selections that include the entity token vs top-k selections.
5. Quantify the confidence drop when the entity token itself is erased vs when non-entity context tokens are erased.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from explainability.attention_extractor import AttentionExtractor
from explainability.erasure_faithfulness_test import (
    get_token_confidence,
    load_dev_sentences_with_entities,
)
from explainability.gradient_attention_attribution import GradientAttentionAttributor
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import load_or_create_multilingual_vocabs


def audit_erasure_design(sample_size: int = 100, seed: int = 42) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    proc_dir = Path("data/processed")
    gold_dir = Path("data/gold")
    checkpoints_dir = Path("checkpoints")
    languages = ("konkani", "assamese", "sinhala")

    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir, languages=languages)

    rng = random.Random(seed)

    print("\n=======================================================")
    print("  AUDIT: Top-k vs Random-k Token Selection Symmetry")
    print("=======================================================\n")

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

        if len(all_instances) > sample_size:
            selected_instances = rng.sample(all_instances, sample_size)
        else:
            selected_instances = all_instances

        print(f"\n--- Language: {lang.upper()} (N = {len(selected_instances)} instances) ---")

        # Track inclusion rates and confidence drops
        # For k in (1, 2, 3)
        entity_in_top_grad = {1: 0, 2: 0, 3: 0}
        entity_in_top_raw = {1: 0, 2: 0, 3: 0}
        entity_in_rand = {1: 0, 2: 0, 3: 0}

        # Track drops when entity IS erased vs when entity is NOT erased
        drops_when_entity_erased = []
        drops_when_context_erased = []

        target_token_attributions_grad = []
        target_token_attributions_raw = []

        for item in selected_instances:
            tokens = item["tokens"]
            q_idx = item["token_idx"]
            seq_len = len(tokens)

            attr_result = attributor.attribute_token(tokens, target_token_idx=q_idx, normalize=True)
            target_tag_id = attr_result["target_tag_id"]
            orig_conf = get_token_confidence(model, extractor, tokens, q_idx, target_tag_id)

            grad_scores = attr_result["attribution_scores"]
            raw_scores = attr_result["raw_attention"]

            target_attr_grad = float(grad_scores[q_idx].item())
            target_attr_raw = float(raw_scores[q_idx].item())
            target_token_attributions_grad.append(target_attr_grad)
            target_token_attributions_raw.append(target_attr_raw)

            grad_ranked_indices = torch.argsort(grad_scores, descending=True).tolist()
            raw_ranked_indices = torch.argsort(raw_scores, descending=True).tolist()

            # Measure drop if target token itself is specifically erased
            tokens_target_erased = list(tokens)
            tokens_target_erased[q_idx] = "<UNK>"
            conf_target_erased = get_token_confidence(model, extractor, tokens_target_erased, q_idx, target_tag_id)
            drops_when_entity_erased.append(max(0.0, orig_conf - conf_target_erased))

            for k in (1, 2, 3):
                if k >= seq_len:
                    continue

                top_k_grad = grad_ranked_indices[:k]
                top_k_raw = raw_ranked_indices[:k]

                if q_idx in top_k_grad:
                    entity_in_top_grad[k] += 1
                if q_idx in top_k_raw:
                    entity_in_top_raw[k] += 1

                # Original random-k pool from erasure_faithfulness_test.py:
                # non_top_pool = [i for i in range(seq_len) if i not in top_k_grad and i not in top_k_raw]
                non_top_pool = [i for i in range(seq_len) if i not in top_k_grad and i not in top_k_raw]
                if len(non_top_pool) >= k:
                    rand_indices = rng.sample(non_top_pool, k)
                else:
                    rand_indices = rng.sample(range(seq_len), min(k, seq_len))

                if q_idx in rand_indices:
                    entity_in_rand[k] += 1

                # If non_top_pool explicitly excludes target token:
                context_pool = [i for i in non_top_pool if i != q_idx]
                if context_pool:
                    ctx_sample = rng.sample(context_pool, min(k, len(context_pool)))
                    tokens_ctx_erased = list(tokens)
                    for idx in ctx_sample:
                        tokens_ctx_erased[idx] = "<UNK>"
                    conf_ctx_erased = get_token_confidence(model, extractor, tokens_ctx_erased, q_idx, target_tag_id)
                    drops_when_context_erased.append(max(0.0, orig_conf - conf_ctx_erased))

        N = len(selected_instances)
        print(f"Target Token Mean Attribution (Grad x Attn): {sum(target_token_attributions_grad)/N:.6f}")
        print(f"Target Token Mean Attribution (Raw Attention): {sum(target_token_attributions_raw)/N:.6f}")
        print(f"Target Token Attribution <= 0.001 rate (Grad x Attn): {sum(1 for a in target_token_attributions_grad if a <= 0.001)/N*100:.1f}%")

        for k in (1, 2, 3):
            print(f"\nk = {k}:")
            print(f"  Entity token in top-{k} Grad x Attn: {entity_in_top_grad[k]}/{N} ({entity_in_top_grad[k]/N*100:.1f}%)")
            print(f"  Entity token in top-{k} Raw Attention: {entity_in_top_raw[k]}/{N} ({entity_in_top_raw[k]/N*100:.1f}%)")
            print(f"  Entity token in random-{k} pool: {entity_in_rand[k]}/{N} ({entity_in_rand[k]/N*100:.1f}%)")

        print(f"\nMean confidence drop when ENTITY TOKEN ITSELF is erased: {sum(drops_when_entity_erased)/len(drops_when_entity_erased):.4f}")
        if drops_when_context_erased:
            print(f"Mean confidence drop when ONLY CONTEXT TOKEN is erased: {sum(drops_when_context_erased)/len(drops_when_context_erased):.4f}")


if __name__ == "__main__":
    audit_erasure_design()
