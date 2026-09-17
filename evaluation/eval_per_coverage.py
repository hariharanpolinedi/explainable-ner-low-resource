"""
Evaluation of PER Coverage Expansion: Manual Cases & Locked Gold-Dev Before vs After.

Tests:
1. Sinhala: "මහාචාර්ය අනුර සිල්වා සහ ආචාර්ය නිමාලි ප්‍රනාන්දු විසින්..."
   Specifically: "අනුර සිල්වා" (PER) and "නිමාලි ප්‍රනාන්දු" (PER).
2. Konkani: "...मुख्यमंत्री डॉ. प्रमोद सावंत आनी केंद्रीय मंत्री श्रीपाद नाईक हांणी..."
   Specifically: "प्रमोद सावंत" (PER) and "श्रीपाद नाईक" (PER).
3. Assamese: "মুখ্যমন্ত্ৰী ড° হিমন্ত বিশ্ব শৰ্মা এই গুৰুত্বপূৰ্ণ প্ৰকল্পটো উদ্বোধন কৰিলে ।"
   Specifically: "হিমন্ত বিশ্ব শৰ্মা" (PER).

Also recomputes strict entity F1 on locked gold-dev (data/gold/{lang}/dev.conll).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from evaluation.seqeval_metrics import compute_seqeval_metrics
from model.ner_model_v4_multilingual import NERMultilingualModel
from preprocessing.normalize import normalize_text
from preprocessing.tokenizer import ScriptTokenizer
from training.multilingual_dataset import (
    MultilingualNERDataset,
    collate_fn_multilingual,
    load_or_create_multilingual_vocabs,
)
from training.train_charcnn import ID_TO_TAG

TEST_CASES = {
    "sinhala": [
        {
            "text": "මහාචාර්ය අනුර සිල්වා සහ ආචාර්ය නිමාලි ප්‍රනාන්දු විසින් පේරාදෙණිය විශ්වවිද්‍යාලයේදී සංවර්ධනය කරන ලද පද්ධතිය ශ්‍රී ලංකාවේ ග්‍රාමීය ප්‍රදේශවල ගොවීන්ට කාලගුණික වෙනස්කම් සහ බෝග රෝග හඳුනා ගැනීමට උපකාරී වේ.",
            "targets": [
                ("අනුර සිල්වා", "PER"),
                ("නිමාලි ප්‍රනාන්දු", "PER"),
            ],
        },
        {
            "text": "ආචාර්ය නිමාලි ප්රනාන්දු විශේෂ පර්යේෂණයක් ඉදිරිපත් කළාය .",
            "targets": [("නිමාලි ප්රනාන්දු", "PER")],
        },
    ],
    "konkani": [
        {
            "text": "पणजींत जाल्ल्या एका विशेष परिषदेंत मुख्यमंत्री डॉ. प्रमोद सावंत आनी केंद्रीय मंत्री श्रीपाद नाईक हांणी गोयाच्या आर्थिक विकासा खातीर नव्या प्रकल्पांची घोशणा केली.",
            "targets": [
                ("प्रमोद सावंत", "PER"),
                ("श्रीपाद नाईक", "PER"),
            ],
        },
    ],
    "assamese": [
        {
            "text": "মুখ্যমন্ত্ৰী ড° হিমন্ত বিশ্ব শৰ্মা এই গুৰুত্বপূৰ্ণ প্ৰকল্পটো উদ্বোধন কৰিলে ।",
            "targets": [("হিমন্ত বিশ্ব শৰ্মা", "PER")],
        },
    ],
}


def load_model(ckpt_path: Path, device: torch.device, w_to_id: dict, c_to_id: dict) -> NERMultilingualModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model = NERMultilingualModel(
        word_vocab_size=len(w_to_id),
        char_vocab_size=len(c_to_id),
        num_tags=9,
        num_languages=3,
        hidden_dim=128,
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def predict_sentence(
    model: NERMultilingualModel,
    text: str,
    lang: str,
    device: torch.device,
    w_to_id: dict,
    c_to_id: dict,
) -> list[tuple[str, str]]:
    lang_ids_map = {"konkani": 0, "assamese": 1, "sinhala": 2}
    lang_id = lang_ids_map[lang]

    tok = ScriptTokenizer(lang)
    tokens = tok.tokenize_words(normalize_text(text, form="NFC", lang=lang))
    if not tokens:
        return []

    word_ids = [w_to_id.get(t, 1) for t in tokens]
    char_ids = [[c_to_id.get(c, 1) for c in t] for t in tokens]

    max_char_len = max((len(c) for c in char_ids), default=1)
    padded_chars = [c + [0] * (max_char_len - len(c)) for c in char_ids]

    w_tensor = torch.tensor([word_ids], dtype=torch.long, device=device)
    c_tensor = torch.tensor([padded_chars], dtype=torch.long, device=device)
    l_tensor = torch.tensor([lang_id], dtype=torch.long, device=device)
    mask = torch.ones((1, len(tokens)), dtype=torch.bool, device=device)

    with torch.no_grad():
        paths = model(w_tensor, c_tensor, lang_ids=l_tensor, mask=mask)
        tags = [ID_TO_TAG[t] for t in paths[0]]

    entities = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            while i < len(tags) and tags[i] == f"I-{etype}":
                i += 1
            span_text = " ".join(tokens[start:i])
            entities.append((span_text, etype))
        else:
            i += 1

    return entities


def evaluate_dev_f1(
    model: NERMultilingualModel,
    dev_path: Path,
    lang: str,
    device: torch.device,
    w_to_id: dict,
    c_to_id: dict,
) -> dict[str, Any]:
    ds = MultilingualNERDataset(
        lang_conll_paths={lang: [dev_path]},
        word_to_id=w_to_id,
        char_to_id=c_to_id,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn_multilingual)

    all_golds = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            paths = model(
                batch["word_ids"].to(device),
                batch["char_ids"].to(device),
                lang_ids=batch["lang_ids"].to(device),
                mask=batch["mask"].to(device),
            )
            for i in range(len(batch["seq_lengths"])):
                slen = batch["seq_lengths"][i]
                g = [ID_TO_TAG[t.item()] for t in batch["tags"][i, :slen]]
                p = [ID_TO_TAG[t] for t in paths[i][:slen]]
                all_golds.append(g)
                all_preds.append(p)

    return compute_seqeval_metrics(all_golds, all_preds)


def run_full_per_evaluation() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    w_to_id, c_to_id = load_or_create_multilingual_vocabs(proc_dir)

    ckpts = {
        "sinhala": {
            "baseline": PROJECT_ROOT / "checkpoints" / "conjunct_fixed_sinhala.pt",
            "per_fixed": PROJECT_ROOT / "checkpoints" / "per_fixed_sinhala.pt",
        },
        "konkani": {
            "baseline": PROJECT_ROOT / "checkpoints" / "selftrained_konkani_best.pt",
            "per_fixed": PROJECT_ROOT / "checkpoints" / "per_fixed_konkani.pt",
        },
        "assamese": {
            "baseline": PROJECT_ROOT / "checkpoints" / "augmented_assamese.pt",
            "per_fixed": PROJECT_ROOT / "checkpoints" / "per_fixed_assamese.pt",
        },
    }

    print("=" * 85)
    print("PER COVERAGE AUDIT: MANUAL TEST-CASES & GOLD-DEV (BEFORE vs AFTER)")
    print("=" * 85)

    for lang in ["sinhala", "konkani", "assamese"]:
        base_p = ckpts[lang]["baseline"]
        per_p = ckpts[lang]["per_fixed"]

        if not per_p.is_file():
            print(f"[{lang.upper()}] Checkpoint {per_p.name} not found.")
            continue

        base_m = load_model(base_p, device, w_to_id, c_to_id)
        per_m = load_model(per_p, device, w_to_id, c_to_id)

        print(f"\n{'='*35} {lang.upper()} {'='*35}")

        # 1. Manual test cases
        print(f"1. Manual Test Cases (Before vs After):")
        for tc in TEST_CASES[lang]:
            b_ents = predict_sentence(base_m, tc["text"], lang, device, w_to_id, c_to_id)
            p_ents = predict_sentence(per_m, tc["text"], lang, device, w_to_id, c_to_id)
            print(f"\n   Input: \"{tc['text'][:65]}...\"")
            print(f"   Targets: {tc['targets']}")
            print(f"   Baseline:  {b_ents}")
            print(f"   Retrained: {p_ents}")

            for target_text, target_type in tc["targets"]:
                b_hit = any(target_text in e[0] for e in b_ents)
                p_hit = any(target_text in e[0] for e in p_ents)
                status = "RECOVERED! 🎉" if (not b_hit and p_hit) else ("STILL MISSED ❌" if not p_hit else "MAINTAINED ✅")
                print(f"     -> Target '{target_text}' ({target_type}): Baseline={'HIT' if b_hit else 'MISS'} | Retrained={'HIT' if p_hit else 'MISS'} -> {status}")

        # 2. Gold-Dev
        dev_p = PROJECT_ROOT / "data" / "gold" / lang / "dev.conll"
        b_dev = evaluate_dev_f1(base_m, dev_p, lang, device, w_to_id, c_to_id)
        p_dev = evaluate_dev_f1(per_m, dev_p, lang, device, w_to_id, c_to_id)

        print(f"\n2. Standard Gold-Dev Metrics ({dev_p.name}):")
        b_ov = b_dev["overall"]["entity_f1"] * 100
        p_ov = p_dev["overall"]["entity_f1"] * 100
        print(f"   Overall Entity F1: Baseline={b_ov:.2f}% | Retrained={p_ov:.2f}% | Delta={p_ov-b_ov:+.2f}pp")

        for etype in ["PER", "LOC", "ORG", "MISC"]:
            bf = b_dev["by_type"][etype]["f1"] * 100
            pf = p_dev["by_type"][etype]["f1"] * 100
            print(f"   {etype:<5} F1: Baseline={bf:6.2f}% | Retrained={pf:6.2f}% | Delta={pf-bf:+6.2f}pp")


if __name__ == "__main__":
    run_full_per_evaluation()
