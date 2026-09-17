"""
PER (Person-Name) Augmentation Generator & Quality Assurance Pipeline.

Generates synthetic training sentences embedding expanded person names and honorific
titles into natural context templates for:
- Konkani
- Sinhala
- Assamese

Enforces Phase-11 discipline:
1. Strict BIO sequence validation (is_valid_bio).
2. Naturalness spot-check logging.
3. Zero pretrained components.
"""

from __future__ import annotations

import random
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

from utils.seed import set_global_seed


def load_gazetteer(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def is_valid_bio(tags: list[str]) -> tuple[bool, str]:
    for i, tag in enumerate(tags):
        if tag.startswith("I-"):
            etype = tag[2:]
            if i == 0 or tags[i - 1] not in (f"B-{etype}", f"I-{etype}"):
                return False, f"Illegal transition at index {i}: {tag} without preceding B/I"
    return True, ""


# Natural sentence templates with [PER] slot per language
TEMPLATES = {
    "konkani": [
        "केंद्रीय मंत्री [PER] हांणी नव्या प्रकल्पांची घोशणा केली .",
        "मुख्यमंत्री डॉ. [PER] हांणी अर्थसंकल्प सादर केलो .",
        "डॉ. [PER] हांणी परिशदेंत म्हत्वाचें भाशण केलें .",
        "संस्थेचे अध्यक्ष श्री. [PER] हांणी पत्रकारांक म्हायती दिली .",
        "[PER] हांणी गोंयच्या संस्कृतीचेर नवें पुस्तक बरयलें .",
        "प्रसिद्ध लेखक [PER] हांकां साहित्याचो पुरस्कार फावो जालो .",
        "कार्यक्रमाक [PER] बाब हजर आशिल्ले .",
        "मंत्री [PER] हांणी लोकांच्या प्रस्नांक जाप दिली .",
        "आमदार [PER] हांणी मतदार संघांत विकास कामांची पाहणी केली .",
        "[PER] हांणी क्रीडा मळार व्हड यश संपादन केलें .",
        "फादर [PER] हांणी समाज सेवेचें कार्य सुरू केलें .",
        "कलाकार [PER] हांणी संगीताची सुंदर मैफल सादर केली .",
    ],
    "sinhala": [
        "මහාචාර්ය [PER] විසින් නව පර්යේෂණ පද්ධතියක් සංවර්ධනය කරන ලදී .",
        "ආචාර්ය [PER] විසින් වෛද්‍ය පර්යේෂණ වාර්තාව ඉදිරිපත් කරන ලදී .",
        "විශ්වවිද්‍යාල ජ්‍යෙෂ්ඨ කථිකාචාර්ය ආචාර්ය [PER] විශේෂ දේශනයක් පැවැත්වීය .",
        "ජනාධිපති [PER] ජාතිය අමතා විශේෂ ප්‍රකාශයක් කළේය .",
        "අගමැති [PER] පාර්ලිමේන්තුවේදී විශේෂ යෝජනාවක් සම්මත කළේය .",
        "ඇමති [PER] සංවර්ධන ව්‍යාපෘතිය නිරීක්ෂණය කිරීමට පැමිණියේය .",
        "[PER] මහතා විසින් මෙම කටයුත්ත සාර්ථකව නිම කරන ලදී .",
        "[PER] මහත්මිය මෙම වැඩසටහන මෙහෙයවනු ලැබීය .",
        "ප්‍රවීණ ක්‍රිකට් ක්‍රීඩක [PER] තරගයේ වීරයා ලෙස සම්මාන ලැබීය .",
        "ප්‍රවීණ සංගීතඥ [PER] විසින් නව ගීත ඇල්බමය එළිදැක්වීය .",
        "[PER] සහ [PER] යන විද්වතුන් මෙම සමුළුවට සහභාගී වූහ .",
        "ගරු [PER] මැතිතුමා විශේෂ අමුත්තා ලෙස එක්විය .",
    ],
    "assamese": [
        "মুখ্যমন্ত্ৰী ড° [PER] এই গুৰুত্বপূৰ্ণ প্ৰকল্পটো উদ্বোধন কৰিলে ।",
        "ড° [PER] অসমৰ ঐতিহাসিক ঐতিহ্য সম্পৰ্কে গৱেষণা কৰিছে ।",
        "ডাঃ [PER] ৰোগীসকলক উন্নত চিকিৎসা সেৱা প্ৰদান কৰিলে ।",
        "মন্ত্ৰী [PER] ৰাজহুৱা সভাত ভাষণ প্ৰদান কৰিলে ।",
        "বিশিষ্ট সাহিত্যিক [PER] অসম সাহিত্য সভাৰ সভাপতি নিৰ্বাচিত হ'ল ।",
        "শিল্পী [PER] এটি সুন্দৰ গীত পৰিৱেশন কৰিলে ।",
        "[PER] মহাশয়ে বিদ্যালয়খনৰ উন্নয়নৰ বাবে দান কৰিলে ।",
        "খেলুৱৈ [PER] ৰাষ্ট্ৰীয় পৰ্যায়ত স্বৰ্ণ পদক লাভ কৰিলে ।",
        "প্ৰাধ্যাপক [PER] ছাত্ৰ-ছাত্ৰীসকলক দিক্‌দৰ্শন দিলে ।",
        "[PER] আৰু [PER] যৌথভাৱে এই গ্ৰন্থখন ৰচনা কৰিলে ।",
        "সাংসদ [PER] সমষ্টিৰ উন্নয়নমূলক কাম-কাজ পৰিদৰ্শন কৰিলে ।",
        "শ্ৰীযুত [PER] সভাত মুখ্য অতিথি হিচাপে উপস্থিত আছিল ।",
    ],
}


def generate_per_augmented_sentences(
    lang: str,
    target_count: int = 400,
    seed: int = 42,
) -> list[tuple[list[str], list[str]]]:
    set_global_seed(seed)
    gaz_path = PROJECT_ROOT / "data" / "gazetteers" / lang / "persons.txt"
    persons = load_gazetteer(gaz_path)
    if not persons:
        raise FileNotFoundError(f"Missing gazetteer persons at: {gaz_path}")

    # Explicit priority entities to guarantee exposure
    priority_names: list[str] = []
    if lang == "konkani":
        priority_names = ["श्रीपाद नाईक", "प्रमोद सावंत", "दिगंबर कामत", "दामोदर मावजो"]
    elif lang == "sinhala":
        priority_names = ["නිමාලි ප්‍රනාන්දු", "නිමාලි ප්රනාන්දු", "අනුර සිල්වා", "මහාචාර්ය අනුර සිල්වා"]
    elif lang == "assamese":
        priority_names = ["হিমন্ত বিশ্ব শৰ্মা", "সৰ্বানন্দ সোণোৱাল", "ভূপেন হাজৰিকা", "লক্ষ্মীনাথ বেজবৰুৱা"]

    templates = TEMPLATES[lang]
    dataset: list[tuple[list[str], list[str]]] = []

    # 1. First, generate sentences for every priority entity across multiple templates
    for pname in priority_names:
        for tmpl in templates[:4]:
            p_tokens = pname.split()
            tmpl_tokens = tmpl.split()

            # Handle templates with multiple [PER]
            new_tokens = []
            new_tags = []
            for tok in tmpl_tokens:
                if tok == "[PER]":
                    new_tokens.extend(p_tokens)
                    new_tags.append("B-PER")
                    new_tags.extend(["I-PER"] * (len(p_tokens) - 1))
                else:
                    new_tokens.append(tok)
                    new_tags.append("O")

            valid, err = is_valid_bio(new_tags)
            assert valid, f"Invalid BIO: {err}"
            dataset.append((new_tokens, new_tags))

    # 2. Fill remainder up to target_count from shuffled persons pool
    random.shuffle(persons)
    p_idx = 0
    while len(dataset) < target_count:
        tmpl = random.choice(templates)
        tmpl_tokens = tmpl.split()

        new_tokens = []
        new_tags = []
        for tok in tmpl_tokens:
            if tok == "[PER]":
                pname = persons[p_idx % len(persons)]
                p_idx += 1
                p_tokens = pname.split()
                new_tokens.extend(p_tokens)
                new_tags.append("B-PER")
                new_tags.extend(["I-PER"] * (len(p_tokens) - 1))
            else:
                new_tokens.append(tok)
                new_tags.append("O")

        valid, err = is_valid_bio(new_tags)
        assert valid, f"Invalid BIO: {err}"
        dataset.append((new_tokens, new_tags))

    return dataset


def save_dataset_conll(
    sentences: list[tuple[list[str], list[str]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for toks, tags in sentences:
            for tok, tag in zip(toks, tags):
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")


def build_all_per_datasets() -> dict[str, Any]:
    print("=" * 80)
    print("GENERATING PER AUGMENTATION DATASETS (Konkani, Sinhala, Assamese)")
    print("=" * 80)

    results = {}
    for lang in ["konkani", "sinhala", "assamese"]:
        data = generate_per_augmented_sentences(lang, target_count=400)
        out_p = PROJECT_ROOT / "data" / "augmented" / lang / "per_augmentation.conll"
        save_dataset_conll(data, out_p)
        print(f"\n[{lang.upper()}] Generated {len(data)} synthetic sentences -> {out_p}")

        # Naturalness spot-check (log 3 sample sentences)
        print(f"   Naturalness Spot-Check Samples:")
        for idx in [0, 50, 100]:
            sent_str = " ".join(data[idx][0])
            tags_str = " ".join(f"{t}:{tag}" for t, tag in zip(data[idx][0], data[idx][1]) if tag != "O")
            print(f"     #{idx+1}: {sent_str}")
            print(f"            Entities: {tags_str}")

        results[lang] = {
            "sentence_count": len(data),
            "output_path": str(out_p),
            "bio_valid": True,
        }

    return results


if __name__ == "__main__":
    build_all_per_datasets()
