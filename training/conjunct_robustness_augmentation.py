"""
Sinhala Training-Time Conjunct Robustness Augmentation.

Generates decomposed (ZWJ-stripped) variants of training sentences for a configurable
fraction (default 25%) of the Sinhala training pool:
- Ensures the neural model (Char-CNN + BiLSTM) learns representations for both
  canonical Unicode conjuncts (e.g. ප්‍ර, ද්‍ය, ශ්‍ර) and real-world decomposed input (ප්ර, ද්ය, ශ්ර).
- Validates BIO sequence integrity and spot-checks naturalness.
- Outputs data/augmented/sinhala/conjunct_robustness.conll.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.sinhala_conjunct_repair import decompose_sinhala_conjuncts
from utils.seed import set_global_seed


def load_conll_sentences(conll_path: Path | str) -> list[list[tuple[str, str]]]:
    sentences: list[list[tuple[str, str]]] = []
    curr: list[tuple[str, str]] = []
    with open(conll_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            l = line.strip()
            if not l:
                if curr:
                    sentences.append(curr)
                    curr = []
            else:
                parts = l.split("\t")
                if len(parts) >= 2:
                    curr.append((parts[0], parts[1]))
                elif len(parts) == 1:
                    curr.append((parts[0], "O"))
    if curr:
        sentences.append(curr)
    return sentences


def save_conll_sentences(sentences: list[list[tuple[str, str]]], out_path: Path | str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for sent in sentences:
            for tok, tag in sent:
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")


def is_valid_bio(tags: list[str]) -> tuple[bool, str]:
    for i, tag in enumerate(tags):
        if tag.startswith("I-"):
            etype = tag[2:]
            if i == 0 or tags[i - 1] not in (f"B-{etype}", f"I-{etype}"):
                return False, f"Illegal transition at index {i}: {tag} without preceding B/I"
    return True, ""


def generate_conjunct_robustness_dataset(
    input_conll: Path | str,
    output_conll: Path | str,
    augmentation_ratio: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    set_global_seed(seed)
    sents = load_conll_sentences(input_conll)
    total_sents = len(sents)

    # Filter sentences that contain at least one conjunct that changes under decomposition
    candidates = []
    for idx, sent in enumerate(sents):
        tokens = [t for t, _ in sent]
        has_conjunct = any(decompose_sinhala_conjuncts(t) != t for t in tokens)
        if has_conjunct:
            candidates.append(idx)

    target_aug_count = int(total_sents * augmentation_ratio)
    selected_indices = random.sample(candidates, min(len(candidates), target_aug_count))

    augmented_sents: list[list[tuple[str, str]]] = []
    spotcheck_samples: list[dict[str, Any]] = []

    for idx in selected_indices:
        orig_sent = sents[idx]
        aug_sent: list[tuple[str, str]] = []
        changed = False
        for tok, tag in orig_sent:
            dec_tok = decompose_sinhala_conjuncts(tok)
            if dec_tok != tok:
                changed = True
            aug_sent.append((dec_tok, tag))

        if changed:
            # Validate BIO integrity
            valid, err = is_valid_bio([tag for _, tag in aug_sent])
            assert valid, f"BIO transition violated during augmentation: {err}"
            augmented_sents.append(aug_sent)

            if len(spotcheck_samples) < 5:
                spotcheck_samples.append({
                    "orig_tokens": [t for t, _ in orig_sent[:10]],
                    "aug_tokens": [t for t, _ in aug_sent[:10]],
                    "tags": [tag for _, tag in orig_sent[:10]],
                })

    save_conll_sentences(augmented_sents, output_conll)

    return {
        "total_source_sentences": total_sents,
        "eligible_conjunct_sentences": len(candidates),
        "augmented_count": len(augmented_sents),
        "augmentation_ratio": len(augmented_sents) / total_sents,
        "output_path": str(output_conll),
        "spotcheck_samples": spotcheck_samples,
    }


if __name__ == "__main__":
    src = PROJECT_ROOT / "data" / "gold" / "sinhala" / "train.conll"
    out = PROJECT_ROOT / "data" / "augmented" / "sinhala" / "conjunct_robustness.conll"
    res = generate_conjunct_robustness_dataset(src, out, augmentation_ratio=0.25)
    print(f"Generated {res['augmented_count']} conjunct-robustness augmented sentences ({res['augmentation_ratio']*100:.1f}%).")
