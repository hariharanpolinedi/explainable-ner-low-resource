"""
Build train_cycle2_corrected_v2.conll with targeted genuine data reinforcement.

Preserves train.conll and train_cycle2_corrected.conll unmodified.
Fixes residual omission in Sentence 377 (মুম্বাইত, মহাৰাষ্ট্ৰৰ, আয়েশা টাকিয়াৰ).
Adds 8 genuine, non-fabricated sentences from the Assamese corpus containing
unrepresented tokens (মুম্বাইত, ৰিজাৰ্ভ, বেংক, নিগম) annotated strictly per
project guidelines.
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weak_supervision.estimate_silver_precision import load_conll_data, extract_entities_from_bio

def build_v2_dataset():
    src_path = PROJECT_ROOT / "data" / "gold" / "assamese" / "train_cycle2_corrected.conll"
    dst_path = PROJECT_ROOT / "data" / "gold" / "assamese" / "train_cycle2_corrected_v2.conll"
    silver_path = PROJECT_ROOT / "data" / "silver" / "assamese" / "train.conll"

    sentences = load_conll_data(src_path)
    print(f"Loaded {len(sentences)} sentences from {src_path.name}")

    # 1. Correct sentence 377 in place
    # Original:
    # গুজৰাটী O, নৃগোষ্ঠীৰ O, আয়েশা O, টাকিয়াৰ O, জন্ম O, হয় O, ভাৰতৰ B-LOC, মহাৰাষ্ট্ৰৰ O, মুম্বাইত O, ...
    s377_toks, s377_tags = sentences[377]
    assert s377_toks[6] == "ভাৰতৰ" and s377_toks[8] == "মুম্বাইত", f"Mismatch in sent 377: {s377_toks}"

    new_s377_tags = list(s377_tags)
    for i, w in enumerate(s377_toks):
        if w == "আয়েশা":
            new_s377_tags[i] = "B-PER"
        elif w == "টাকিয়াৰ":
            new_s377_tags[i] = "I-PER"
        elif w == "মহাৰাষ্ট্ৰৰ":
            new_s377_tags[i] = "B-LOC"
        elif w == "মুম্বাইত":
            new_s377_tags[i] = "B-LOC"

    sentences[377] = (s377_toks, new_s377_tags)
    print(f"Sentence 377 corrected: {list(zip(s377_toks, new_s377_tags))}")

    # 2. Extract genuine corpus sentences from silver pool and annotate per guidelines
    silver_sents = load_conll_data(silver_path)

    # We select 8 genuine sentences containing S200 key vocabulary:
    # 1279: মুম্বাইত (B-LOC)
    # 1323: ভাৰতীয় (B-LOC), ৰিজাৰ্ভ (O), বেংকৰ (B-ORG)
    # 2569: নিগমৰ (B-ORG), মুম্বাইত (B-LOC)
    # 2862: ভাৰতীয় (B-LOC), ৰিজাৰ্ভ (O), বেংকে (B-ORG)
    # 3509: জে জে স্কুল অৱ আৰ্ট (B-ORG I-ORG I-ORG I-ORG I-ORG), মুম্বাইত (B-LOC)
    # 3995: মুম্বাইতে (B-LOC), ভাৰতৰ (B-LOC)
    # 5662: ষ্টেট (O), বেংক (B-ORG), ৰিজাৰ্ভ (O), বেংকে (B-ORG), পাকিস্তানৰ (B-LOC)
    # 5775: ভাৰতৰ (B-LOC), মুম্বাইত (B-LOC)
    additions = []

    # Sent 1279
    toks1279, _ = silver_sents[1279]
    tags1279 = ["O"] * len(toks1279)
    for i, w in enumerate(toks1279):
        if w == "মুম্বাইত":
            tags1279[i] = "B-LOC"
    additions.append((toks1279, tags1279))

    # Sent 1323
    toks1323, _ = silver_sents[1323]
    tags1323 = ["O"] * len(toks1323)
    for i, w in enumerate(toks1323):
        if w == "ভাৰতীয়":
            tags1323[i] = "B-LOC"
        elif w == "বেংকৰ":
            tags1323[i] = "B-ORG"
    additions.append((toks1323, tags1323))

    # Sent 2569
    toks2569, _ = silver_sents[2569]
    tags2569 = ["O"] * len(toks2569)
    for i, w in enumerate(toks2569):
        if w == "নিগমৰ":
            tags2569[i] = "B-ORG"
        elif w == "মুম্বাইত":
            tags2569[i] = "B-LOC"
    additions.append((toks2569, tags2569))

    # Sent 2862
    toks2862, _ = silver_sents[2862]
    tags2862 = ["O"] * len(toks2862)
    for i, w in enumerate(toks2862):
        if w == "ভাৰতীয়":
            tags2862[i] = "B-LOC"
        elif w == "বেংকে":
            tags2862[i] = "B-ORG"
    additions.append((toks2862, tags2862))

    # Sent 3509
    toks3509, _ = silver_sents[3509]
    tags3509 = ["O"] * len(toks3509)
    # জে জে স্কুল অৱ আৰ্ট
    for i, w in enumerate(toks3509):
        if w == "জে" and i + 4 < len(toks3509) and toks3509[i+1] == "জে" and toks3509[i+2] == "স্কুল":
            tags3509[i] = "B-ORG"
            tags3509[i+1] = "I-ORG"
            tags3509[i+2] = "I-ORG"
            tags3509[i+3] = "I-ORG"
            tags3509[i+4] = "I-ORG"
        elif w == "মুম্বাইত":
            tags3509[i] = "B-LOC"
    additions.append((toks3509, tags3509))

    # Sent 3995
    toks3995, _ = silver_sents[3995]
    tags3995 = ["O"] * len(toks3995)
    for i, w in enumerate(toks3995):
        if w == "মুম্বাইতে":
            tags3995[i] = "B-LOC"
        elif w == "ভাৰতৰ":
            tags3995[i] = "B-LOC"
    additions.append((toks3995, tags3995))

    # Sent 5662
    toks5662, _ = silver_sents[5662]
    tags5662 = ["O"] * len(toks5662)
    for i, w in enumerate(toks5662):
        if w == "ষ্টেট":
            tags5662[i] = "O"
        elif w == "বেংক" and i > 0 and toks5662[i-1] == "ষ্টেট":
            tags5662[i] = "B-ORG"
        elif w == "বেংকে":
            tags5662[i] = "B-ORG"
        elif w == "পাকিস্তানৰ":
            tags5662[i] = "B-LOC"
    additions.append((toks5662, tags5662))

    # Sent 5775
    toks5775, _ = silver_sents[5775]
    tags5775 = ["O"] * len(toks5775)
    for i, w in enumerate(toks5775):
        if w == "ভাৰতৰ":
            tags5775[i] = "B-LOC"
        elif w == "মুম্বাইত":
            tags5775[i] = "B-LOC"
    additions.append((toks5775, tags5775))

    all_sents = sentences + additions
    print(f"Total sentences for v2: {len(all_sents)} ({len(sentences)} base + {len(additions)} genuine additions)")

    # Write output with TAB separator
    with open(dst_path, "w", encoding="utf-8") as f:
        for toks, tags in all_sents:
            for w, t in zip(toks, tags):
                f.write(f"{w}\t{t}\n")
            f.write("\n")

    print(f"Successfully saved {dst_path}")

if __name__ == "__main__":
    build_v2_dataset()
