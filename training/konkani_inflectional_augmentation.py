"""
Konkani Morphological/Inflectional Augmentation Generator & Retraining (Cycle 2).

Directly addresses the confirmed root cause from Step 1:
- Real-world inflected forms (e.g. पणजींत, गोयाच्या, सावंतांनी, वास्कोक, म्हापश्यांत)
  are systematically OOV in static word embeddings and trigger 100% false-negative collapse.
- Generates synthetic training instances containing productive case-inflected forms:
  * Locative: -ांत, -ंत
  * Genitive: -च्या, -ाच्या, -चो, -ची, -चे
  * Dative: -क, -ाक
  * Ergative/Instrumental: -ांनी, -ान, -ेन
  along with confirmed irregular stem alternations (गोंय -> गोयाच्या/गोयाक/गोयांत, म्हापसा -> म्हापश्यांत, etc.).
- Follows Phase 11 naturalness and BIO validity checks.
- Retrains starting strictly from checkpoints/improved_konkani_v2.pt with divergence gate enforcement.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.phase5_targeted_interventions import (
    evaluate_model_metrics,
    load_or_create_multilingual_vocabs,
)
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.checkpoint_divergence_gate import save_patched_checkpoint_with_gate
from training.multilingual_dataset import (
    MultilingualNERDataset,
    collate_fn_multilingual,
)
from utils.seed import set_global_seed
from weak_supervision.estimate_silver_precision import (
    extract_entities_from_bio,
    load_conll_data,
)

# Explicit stem dictionary for irregular Konkani entities
KONKANI_IRREGULARS: dict[str, dict[str, str]] = {
    "गोंय": {
        "locative": "गोयांत",
        "genitive": "गोयाच्या",
        "genitive_masc": "गोयाचो",
        "genitive_fem": "गोयाची",
        "dative": "गोयाक",
        "instrumental": "गोयान",
    },
    "गोवा": {
        "locative": "गोयांत",
        "genitive": "गोयाच्या",
        "dative": "गोयाक",
        "instrumental": "गोयान",
    },
    "पणजी": {
        "locative": "पणजींत",
        "genitive": "पणजीच्या",
        "dative": "पणजीक",
        "instrumental": "पणजीन",
    },
    "म्हापसा": {
        "locative": "म्हापश्यांत",
        "genitive": "म्हापश्याच्या",
        "dative": "म्हापशाक",
        "instrumental": "म्हापशान",
    },
    "मडगांव": {
        "locative": "मडगांवांत",
        "genitive": "मडगांवाच्या",
        "dative": "मडगांवाक",
        "instrumental": "मडगांवान",
    },
    "वास्को": {
        "locative": "वास्कोक",
        "genitive": "वास्कोच्या",
        "dative": "वास्कोक",
        "instrumental": "वास्कोन",
    },
    "कुडचडें": {
        "locative": "कुडचड्यांत",
        "genitive": "कुडचड्याच्या",
        "dative": "कुडचड्याक",
        "instrumental": "कुडचड्यान",
    },
    "भारत": {
        "locative": "भारतांत",
        "genitive": "भारताच्या",
        "dative": "भारताक",
        "instrumental": "भारतान",
    },
    "दिल्ली": {
        "locative": "दिल्लींत",
        "genitive": "दिल्लीच्या",
        "dative": "दिल्लीक",
        "instrumental": "दिल्लीन",
    },
    "मुंबय": {
        "locative": "मुंबयंत",
        "genitive": "मुंबयच्या",
        "dative": "मुंबयक",
        "instrumental": "मुंबयन",
    },
}

# Personal name ergative suffixes (confirmed from Step 1)
# सावंतांनी, नायकांनी, कामातींनी, आंबेडकरांनी, पर्रीकारांनी, मावज्यांनी
KONKANI_PER_ERGATIVE: dict[str, str] = {
    "सावंत": "सावंतांनी",
    "नाईक": "नायकांनी",
    "कामत": "कामातींनी",
    "आंबेडकर": "आंबेडकरांनी",
    "पर्रीकार": "पर्रीकारांनी",
    "मावजो": "मावज्यांनी",
    "बोरकार": "बोरकारांनी",
    "गोंयबाब": "गोंयबाबांनी",
    "केळेकार": "केळेकारांनी",
    "सरदेसाय": "सरदेसायांनी",
    "राणे": "राण्यांनी",
    "प्रभू": "प्रभूंनी",
    "शेट": "शेटांनी",
}


def inflect_konkani_entity(lemma: str, etype: str) -> list[tuple[str, str]]:
    """
    Generate productive case-inflected forms for Konkani entity lemma.
    Returns list of (inflected_form, case_name).
    """
    results: list[tuple[str, str]] = []

    # 1. Check irregular tables
    if lemma in KONKANI_IRREGULARS:
        for cname, iform in KONKANI_IRREGULARS[lemma].items():
            results.append((iform, cname))

    if etype == "PER":
        if lemma in KONKANI_PER_ERGATIVE:
            results.append((KONKANI_PER_ERGATIVE[lemma], "ergative"))
        else:
            # Rule for ergative plural/honorific: add -ांनी or -ंनी
            if lemma.endswith("ी") or lemma.endswith("ू"):
                results.append((lemma + "ंनी", "ergative"))
            elif lemma.endswith("ो"):
                results.append((lemma[:-1] + "्यांनी", "ergative"))
            elif lemma.endswith("ा"):
                results.append((lemma + "ंनी", "ergative"))
            else:
                results.append((lemma + "ांनी", "ergative"))

            # Genitive for person
            if lemma.endswith("ो"):
                results.append((lemma[:-1] + "्याचो", "genitive_masc"))
                results.append((lemma[:-1] + "्याची", "genitive_fem"))
                results.append((lemma[:-1] + "्याच्या", "genitive_oblique"))
            else:
                results.append((lemma + "ाचो", "genitive_masc"))
                results.append((lemma + "ाची", "genitive_fem"))
                results.append((lemma + "ाच्या", "genitive_oblique"))

    elif etype in ("LOC", "ORG"):
        if lemma not in KONKANI_IRREGULARS:
            # Regular locative: -ांत or -ंत
            if lemma.endswith("ी") or lemma.endswith("ू") or lemma.endswith("ो"):
                loc = lemma + "ंत"
                dat = lemma + "क"
                gen = lemma + "च्या"
            else:
                loc = lemma + "ांत"
                dat = lemma + "ाक"
                gen = lemma + "ाच्या"
            results.append((loc, "locative"))
            results.append((dat, "dative"))
            results.append((gen, "genitive"))
            results.append((lemma + "चो", "genitive_masc"))
            results.append((lemma + "ची", "genitive_fem"))

    return results


def is_valid_bio(tags: list[str]) -> tuple[bool, str]:
    for i, tag in enumerate(tags):
        if tag.startswith("I-"):
            etype = tag[2:]
            if i == 0 or tags[i - 1] not in (f"B-{etype}", f"I-{etype}"):
                return False, f"Illegal transition at index {i}: {tag} without preceding B/I"
    return True, ""


# Sentence templates reflecting natural Konkani syntactic patterns
KONKANI_TEMPLATES: dict[str, list[str]] = {
    "LOC_LOCATIVE": [
        "[ENT] जाल्ल्या एका विशेष परिषदेंत म्हत्वाचे निर्णय घेतले .",
        "हो म्हत्वाचो आंतरराष्ट्रीय महोत्सव [ENT] आयोजीत केल्लो .",
        "नवो प्रकल्प [ENT] सुरू जावचो आसा .",
        "लोकांनी [ENT] एकवट करून विकास कामांची मागणी केली .",
        "फाटल्या वर्सा [ENT] जायते देशी आनी विदेशी पर्यटक पावले .",
        "शासकीय कार्यालय [ENT] सुरू करपाचो निर्णय जाला .",
        "कला आनी संस्कृती प्रदर्शनाचें आयोजन [ENT] केलें .",
        "अनेक नामनेचे कलाकार [ENT] एकठांय आयले .",
        "उद्येगाक चालना दिवपा खातीर [ENT] नवें केंद्र उबारलें .",
        "सगळ्यांत व्हड प्रदर्शन [ENT] भरिल्लें .",
    ],
    "LOC_DATIVE": [
        "काल मुखावेलो [ENT] एका समारंभाक गेलो .",
        "मंत्री [ENT] उद्घाटना खातीर पावले .",
        "खूप पर्यटक [ENT] भेट दितात .",
        "अधिकारी वर्गाक [ENT] धाडपाचो निर्णय घेतलो .",
        "ते कामा खातीर [ENT] परत आयले .",
        "लोकांनी [ENT] वचून आपल्यो समस्या मांडत्यो .",
    ],
    "LOC_GENITIVE": [
        "[ENT] आर्थिक विकासा खातीर पर्यटन म्हत्वाचें आसा .",
        "[ENT] संस्कृती आनी परंपरा खूब समृद्ध आसा .",
        "[ENT] जनतेन हो म्हत्वाचो निर्णय मानून घेतलो .",
        "इतिहासांत [ENT] घडणूक खूब म्हत्वाची मानतात .",
        "[ENT] निसर्गसौंदर्य पळोवपाक लोक येतात .",
        "[ENT] प्रसिद्ध देवळांची माहिती मेळ्ळी .",
    ],
    "PER_ERGATIVE": [
        "[ENT] नवी येवजण जाहीर केली .",
        "[ENT] कें द्रांत महत्वाची बसका घेतली .",
        "[ENT] मडगांवांत कार्यकर्त्यांक मार्गदर्शन केलें .",
        "[ENT] संविधानाची रचना केली .",
        "[ENT] साहित्यांत मोलादीक योगदान दिलें .",
        "[ENT] पत्रकार परिशदेंत स्पश्टीकरण दिलें .",
        "[ENT] उलयताना समाजसेवेचो संदेश दिलो .",
        "[ENT] लोकांच्यो अडचणी जाणून घेतल्यो .",
        "[ENT] विधानसभेत नवो विधेयक मांडलो .",
        "[ENT] पुस्तकाचें विमोचन केलें .",
    ],
    "PER_GENITIVE": [
        "[ENT] मार्गदर्शना खाला नवो प्रकल्प पुराय जालो .",
        "[ENT] साहित्याची तोखणाय सगळ्यांनी केली .",
        "[ENT] विचार तरणाट्यांक प्रेरणा दितात .",
        "[ENT] अध्यक्षते खाला ही बसका पार पडली .",
        "[ENT] स्मरणार्थ हो पुरस्कार दिला .",
    ],
}


def generate_konkani_inflectional_dataset(
    gold_train_path: Path,
    output_path: Path,
    target_count: int = 500,
    seed: int = 42,
) -> int:
    set_global_seed(seed)
    train_sents = load_conll_data(gold_train_path)

    # Extract all unique entities from gold train
    extracted_entities: dict[str, set[str]] = {
        "PER": set(),
        "LOC": set(),
        "ORG": set(),
        "MISC": set(),
    }

    for toks, tags in train_sents:
        spans = extract_entities_from_bio(tags)
        for etype, s, e in spans:
            mention = " ".join(toks[s:e])
            if mention in extracted_entities:
                extracted_entities[etype].add(mention)

    # Add gazetteer / confirmed Step 1 entities
    step1_locs = ["पणजी", "गोवा", "गोंय", "वास्को", "म्हापसा", "मडगांव", "भारत", "दिल्ली", "मुंबय", "कुडचडें", "फोंडें", "सांखळी", "पेडणें", "काणकोण"]
    step1_pers = ["प्रमोद सावंत", "सावंत", "श्रीपाद नाईक", "नाईक", "दिगंबर कामत", "कामत", "आंबेडकर", "मनोहर पर्रीकार", "पर्रीकार", "दामोदर मावजो", "मावजो", "बाकीबाब बोरकार", "शणै गोंयबाब"]

    for l in step1_locs:
        extracted_entities["LOC"].add(l)
    for p in step1_pers:
        extracted_entities["PER"].add(p)

    generated_sentences: list[tuple[list[str], list[str]]] = []

    # Generate LOC sentences
    for loc_lemma in extracted_entities["LOC"]:
        # Only single-word loc lemmas or last token for multi-word
        parts = loc_lemma.split()
        prefix = " ".join(parts[:-1]) if len(parts) > 1 else ""
        head = parts[-1]

        inflections = inflect_konkani_entity(head, "LOC")
        for iform, ctype in inflections:
            full_inf = f"{prefix} {iform}".strip()
            inf_toks = full_inf.split()

            # Choose template
            if "locative" in ctype:
                tmpl = random.choice(KONKANI_TEMPLATES["LOC_LOCATIVE"])
            elif "dative" in ctype:
                tmpl = random.choice(KONKANI_TEMPLATES["LOC_DATIVE"])
            else:
                tmpl = random.choice(KONKANI_TEMPLATES["LOC_GENITIVE"])

            # Substitute into template
            tmpl_toks = tmpl.split()
            slot_idx = tmpl_toks.index("[ENT]")

            sent_toks = tmpl_toks[:slot_idx] + inf_toks + tmpl_toks[slot_idx + 1:]
            sent_tags = ["O"] * len(tmpl_toks[:slot_idx])
            for i in range(len(inf_toks)):
                sent_tags.append("B-LOC" if i == 0 else "I-LOC")
            sent_tags.extend(["O"] * len(tmpl_toks[slot_idx + 1:]))

            ok, err = is_valid_bio(sent_tags)
            if ok:
                generated_sentences.append((sent_toks, sent_tags))

    # Generate PER sentences
    for per_lemma in extracted_entities["PER"]:
        parts = per_lemma.split()
        prefix = " ".join(parts[:-1]) if len(parts) > 1 else ""
        head = parts[-1]

        inflections = inflect_konkani_entity(head, "PER")
        for iform, ctype in inflections:
            full_inf = f"{prefix} {iform}".strip()
            inf_toks = full_inf.split()

            if "ergative" in ctype:
                tmpl = random.choice(KONKANI_TEMPLATES["PER_ERGATIVE"])
            else:
                tmpl = random.choice(KONKANI_TEMPLATES["PER_GENITIVE"])

            tmpl_toks = tmpl.split()
            slot_idx = tmpl_toks.index("[ENT]")

            sent_toks = tmpl_toks[:slot_idx] + inf_toks + tmpl_toks[slot_idx + 1:]
            sent_tags = ["O"] * len(tmpl_toks[:slot_idx])
            for i in range(len(inf_toks)):
                sent_tags.append("B-PER" if i == 0 else "I-PER")
            sent_tags.extend(["O"] * len(tmpl_toks[slot_idx + 1:]))

            ok, err = is_valid_bio(sent_tags)
            if ok:
                generated_sentences.append((sent_toks, sent_tags))

    # Shuffle and trim to target_count
    random.seed(seed)
    random.shuffle(generated_sentences)
    if len(generated_sentences) > target_count:
        generated_sentences = generated_sentences[:target_count]

    # Write out CoNLL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for toks, tags in generated_sentences:
            for tok, tag in zip(toks, tags):
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")

    print(f"[KONKANI AUGMENTATION] Generated {len(generated_sentences)} high-quality inflected sentences -> {output_path}")
    print("[KONKANI AUGMENTATION] Sample spot-checks:")
    for s_i, (toks, tags) in enumerate(generated_sentences[:5]):
        labeled = " ".join(f"{t}/{g}" for t, g in zip(toks, tags) if g != "O")
        print(f"  Sample {s_i+1}: {' '.join(toks[:12])}... | Entities: {labeled}")

    return len(generated_sentences)


def train_konkani_augmentation_candidate(
    baseline_ckpt_path: Path,
    output_ckpt_path: Path,
    aug_conll_path: Path,
    epochs: int = 5,
    lr: float = 5e-5,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Retrains Konkani model from checkpoints/improved_konkani_v2.pt
    incorporating gold train (2x oversampled) + inflectional augmentation.
    """
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    gold_dir = PROJECT_ROOT / "data" / "gold" / "konkani"

    print("\n" + "=" * 80)
    print(f"KONKANI CANDIDATE 1: INFLECTIONAL AUGMENTATION RETRAINING (Device: {device})")
    print(f"Starting Baseline Checkpoint: {baseline_ckpt_path.name}")
    print("=" * 80)

    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)

    train_paths = [
        gold_dir / "train.conll",
        gold_dir / "train.conll",
        aug_conll_path,
    ]

    train_ds = MultilingualNERDataset(
        lang_conll_paths={"konkani": train_paths},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )
    dev_path = gold_dir / "dev.conll"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_multilingual,
    )

    # Initialize model
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

    # Load baseline checkpoint weights
    ckpt = torch.load(baseline_ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)

    # Snapshot baseline state for divergence gate
    baseline_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Baseline evaluation
    base_eval = evaluate_model_metrics(model, dev_path, "konkani", word_to_id, char_to_id, device)
    print(f"Baseline Gold-Dev F1: {base_eval['entity_f1']:.2f}% (P: {base_eval['entity_precision']:.2f}%, R: {base_eval['entity_recall']:.2f}%)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_cand_state: dict[str, torch.Tensor] | None = None
    best_eval: dict[str, Any] | None = None
    best_f1 = -1.0

    print(f"Retraining on {len(train_ds):,} sentences for {epochs} epochs...")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            loss = model(word_ids, char_ids, lang_ids=lang_ids, tags=tags, mask=mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        ep_eval = evaluate_model_metrics(model, dev_path, "konkani", word_to_id, char_to_id, device)
        cur_f1 = ep_eval["entity_f1"]
        print(f"  Epoch {ep:2d}/{epochs:2d} - Loss: {avg_loss:.4f} | Dev F1: {cur_f1:.2f}% (P: {ep_eval['entity_precision']:.2f}%, R: {ep_eval['entity_recall']:.2f}%)")

        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_eval = ep_eval
            best_cand_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_cand_state is not None

    metadata = {
        "candidate": "cycle2_konkani_augmentation",
        "epochs_trained": epochs,
        "best_dev_f1": best_f1,
        "dev_eval": best_eval,
        "seed": seed,
    }

    # Divergence gate enforcement
    passed, div = save_patched_checkpoint_with_gate(
        candidate_state=best_cand_state,
        baseline_state=baseline_state,
        output_path=output_ckpt_path,
        metadata=metadata,
        min_pct_changed=1.0,
        min_l2_diff=0.01,
    )

    if not passed:
        raise RuntimeError(f"Divergence gate FAILED for {output_ckpt_path.name}: {div}")

    return {
        "candidate": "cycle2_konkani_augmentation",
        "best_dev_f1": best_f1,
        "dev_eval": best_eval,
        "divergence": div,
        "checkpoint_path": str(output_ckpt_path),
    }


if __name__ == "__main__":
    aug_path = PROJECT_ROOT / "data" / "augmented" / "konkani" / "inflectional_cycle2.conll"
    gold_train = PROJECT_ROOT / "data" / "gold" / "konkani" / "train.conll"
    generate_konkani_inflectional_dataset(gold_train, aug_path, target_count=500)

    base_ckpt = PROJECT_ROOT / "checkpoints" / "improved_konkani_v2.pt"
    out_ckpt = PROJECT_ROOT / "checkpoints" / "cycle2_konkani_augmentation.pt"
    train_konkani_augmentation_candidate(base_ckpt, out_ckpt, aug_path, epochs=5, lr=5e-5)
