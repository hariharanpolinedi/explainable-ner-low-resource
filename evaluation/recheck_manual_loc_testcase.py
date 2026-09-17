"""
Manual Test-Case Re-Check: Konkani Location Inflection Verification.

Re-evaluates the exact real-world Konkani sentence:
"पणजींत जाल्ल्या एका विशेष परिषदेंत मुख्यमंत्री डॉ. प्रमोद सावंत आनी केंद्रीय मंत्री श्रीपाद नाईक हांणी गोयाच्या आर्थिक विकासा खातीर नव्या प्रकल्पांची घोशणा केली."

Evaluates both:
1. Genuine Baseline Checkpoint: checkpoints/selftrained_konkani_best.pt
2. LOC-Patched Checkpoint: checkpoints/loc_patched_konkani.pt

Generates reports/loc_patch_manual_test_recheck.md with explicit pass/fail detection results.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from model.ner_model_v4_multilingual import NERMultilingualModel
from training.multilingual_dataset import LANG_TO_ID, load_or_create_multilingual_vocabs
from training.train_charcnn import ID_TO_TAG


TEST_SENTENCE = (
    "पणजींत जाल्ल्या एका विशेष परिषदेंत मुख्यमंत्री डॉ. प्रमोद सावंत आनी "
    "केंद्रीय मंत्री श्रीपाद नाईक हांणी गोयाच्या आर्थिक विकासा खातीर नव्या प्रकल्पांची घोशणा केली."
)


def load_model(checkpoint_path: Path | str, proc_dir: Path | str, device: torch.device) -> tuple[NERMultilingualModel, dict[str, int], dict[str, int]]:
    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)
    model = NERMultilingualModel(
        word_vocab_size=len(word_to_id),
        char_vocab_size=len(char_to_id),
        num_tags=9,
        num_languages=3,
        word_embedding_dim=100,
        char_embedding_dim=50,
        lang_embedding_dim=16,
        filter_widths=(3, 4, 5),
        filters_per_width=30,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_attention_blocks=1,
        attention_dropout=0.1,
        crf_config="shared",
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    return model, word_to_id, char_to_id


def predict_sentence(
    model: NERMultilingualModel,
    tokens: list[str],
    lang: str,
    word_to_id: dict[str, int],
    char_to_id: dict[str, int],
    device: torch.device,
    max_char_len: int = 16,
) -> list[str]:
    w_ids = [word_to_id.get(t, word_to_id.get("<UNK>", 1)) for t in tokens]
    c_ids: list[list[int]] = []
    for t in tokens:
        c_seq = [char_to_id.get(c, char_to_id.get("<UNK>", 1)) for c in t[:max_char_len]]
        if len(c_seq) < max_char_len:
            c_seq = c_seq + [char_to_id.get("<PAD>", 0)] * (max_char_len - len(c_seq))
        c_ids.append(c_seq)

    w_tensor = torch.tensor([w_ids], dtype=torch.long, device=device)
    c_tensor = torch.tensor([c_ids], dtype=torch.long, device=device)
    l_tensor = torch.tensor([LANG_TO_ID[lang]], dtype=torch.long, device=device)
    m_tensor = torch.tensor([[True] * len(tokens)], dtype=torch.bool, device=device)

    with torch.no_grad():
        tag_ids = model(w_tensor, c_tensor, lang_ids=l_tensor, mask=m_tensor)[0]

    return [ID_TO_TAG.get(i, "O") for i in tag_ids]


def run_manual_recheck() -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    base_ckpt = PROJECT_ROOT / "checkpoints" / "selftrained_konkani_best.pt"
    patch_ckpt = PROJECT_ROOT / "checkpoints" / "loc_patched_konkani.pt"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    tokens = TEST_SENTENCE.split()

    print(f"\n--- Loading Baseline Checkpoint: {base_ckpt.name} ---")
    base_model, word_to_id, char_to_id = load_model(base_ckpt, proc_dir, device)
    base_tags = predict_sentence(base_model, tokens, "konkani", word_to_id, char_to_id, device)

    print(f"\n--- Loading LOC-Patched Checkpoint: {patch_ckpt.name} ---")
    patch_model, _, _ = load_model(patch_ckpt, proc_dir, device)
    patch_tags = predict_sentence(patch_model, tokens, "konkani", word_to_id, char_to_id, device)

    # Check target entities: 'पणजींत' (index 0) and 'गोयाच्या' (index 15)
    panaji_base = base_tags[0]
    panaji_patch = patch_tags[0]
    goa_base = base_tags[15]
    goa_patch = patch_tags[15]

    panaji_fixed = panaji_patch in ("B-LOC", "I-LOC")
    goa_fixed = goa_patch in ("B-LOC", "I-LOC")

    print("\n" + "=" * 80)
    print("MANUAL TEST-CASE INFERENCE COMPARISON (KONKANI)")
    print("=" * 80)
    print(f"{'Idx':<4} {'Token':<18} {'Baseline Pred':<16} {'LOC-Patched Pred':<16} {'Target'}")
    print("-" * 80)
    for idx, (t, bt, pt) in enumerate(zip(tokens, base_tags, patch_tags)):
        is_target = " <<< TARGET LOC" if idx in (0, 15) else ""
        print(f"{idx:<4} {t:<18} {bt:<16} {pt:<16} {is_target}")

    print("\nTarget Entity Recovery Summary:")
    print(f"1. पणजींत (Panaji + locative '-ंत'): Baseline = {panaji_base} -> LOC-Patched = {panaji_patch} [{'FIXED' if panaji_fixed else 'MISSED'}]")
    print(f"2. गोयाच्या (Goa + genitive '-च्या'): Baseline = {goa_base} -> LOC-Patched = {goa_patch} [{'FIXED' if goa_fixed else 'MISSED'}]")

    # Generate reports/loc_patch_manual_test_recheck.md
    report_file = reports_dir / "loc_patch_manual_test_recheck.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Manual Test-Case Re-Check: Konkani Location Inflection Verification\n\n")
        f.write("**Date**: 2026-09-17  \n")
        f.write("**Auditor**: Senior ML/NLP Engineer  \n")
        f.write("**Test Case**: Authentic Konkani Government Press Release Sentence  \n\n")
        f.write("```text\n")
        f.write(TEST_SENTENCE + "\n")
        f.write("```\n\n")
        f.write("---\n\n")
        f.write("## 1. Ground Truth Annotation vs Model Predictions\n\n")
        f.write("| Token Index | Token | Ground Truth Tag | Baseline (`selftrained_best`) | LOC-Patched (`loc_patched`) | Status |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: |\n")

        gold_tags_map = {
            0: "B-LOC",   # पणजींत
            5: "O",       # मुख्यमंत्री
            6: "O",       # डॉ.
            7: "B-PER",   # प्रमोद
            8: "I-PER",   # सावंत
            9: "O",       # आनी
            10: "O",      # केंद्रीय
            11: "O",      # मंत्री
            12: "B-PER",  # श्रीपाद
            13: "I-PER",  # नाईक
            14: "O",      # हांणी
            15: "B-LOC",  # गोयाच्या
        }

        for idx, (t, bt, pt) in enumerate(zip(tokens, base_tags, patch_tags)):
            gt = gold_tags_map.get(idx, "O")
            status = "MATCH" if pt == gt else ("RECOVERED" if pt.startswith("B-LOC") and gt.startswith("B-LOC") else "MISSED")
            if idx in (0, 15):
                f.write(f"| **{idx}** | **{t}** | **{gt}** | `{bt}` | **`{pt}`** | **{status}** |\n")
            else:
                f.write(f"| {idx} | {t} | {gt} | `{bt}` | `{pt}` | {status} |\n")

        f.write("\n---\n\n")
        f.write("## 2. Recovery Analysis for पणजींत and गोयाच्या\n\n")
        f.write(f"- **Token 0 (`पणजींत`)**: Baseline predicted `{panaji_base}`. LOC-Patched predicted **`{panaji_patch}`**.\n")
        f.write(f"  - *Result*: {'SUCCESSFULLY RECOVERED' if panaji_fixed else 'STILL MISSED'}.\n")
        f.write(f"- **Token 15 (`गोयाच्या`)**: Baseline predicted `{goa_base}`. LOC-Patched predicted **`{goa_patch}`**.\n")
        f.write(f"  - *Result*: {'SUCCESSFULLY RECOVERED' if goa_fixed else 'STILL MISSED'}.\n\n")
        f.write("---\n\n")
        f.write("## 3. Findings\n\n")
        if panaji_fixed and goa_fixed:
            f.write("Both previously missed location entities (`पणजींत` and `गोयाच्या`) are now accurately detected by the retrained checkpoint.\n")
        elif panaji_fixed or goa_fixed:
            f.write("Partial recovery achieved: one of the two inflected location entities is now detected.\n")
        else:
            f.write("Neither inflected entity was recovered by the retrained checkpoint.\n")

    print(f"\n[COMPLETE] Wrote manual test report to: {report_file}")
    return {
        "tokens": tokens,
        "base_tags": base_tags,
        "patch_tags": patch_tags,
        "panaji_fixed": panaji_fixed,
        "goa_fixed": goa_fixed,
    }


if __name__ == "__main__":
    run_manual_recheck()
