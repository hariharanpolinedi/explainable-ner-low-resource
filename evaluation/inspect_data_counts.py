"""
Inspect and verify exact sentence counts across all data sources:
Gold, Silver, Pseudo, and Augmented.
"""
from pathlib import Path

def count_conll_sentences(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    in_sent = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if in_sent:
                    count += 1
                    in_sent = False
            else:
                in_sent = True
        if in_sent:
            count += 1
    return count

languages = ["konkani", "assamese", "sinhala"]

print("=== EXACT SENTENCE COUNTS PER SOURCE ===")
for lang in languages:
    gold_train = Path(f"data/gold/{lang}/train.conll")
    gold_dev = Path(f"data/gold/{lang}/dev.conll")
    gold_test = Path(f"data/gold/{lang}/test.conll")
    silver = Path(f"data/silver/{lang}/train.conll")
    
    pseudo_r1 = Path(f"data/pseudo/{lang}/round_1.conll")
    pseudo_r2 = Path(f"data/pseudo/{lang}/round_2.conll")
    
    aug_entity = Path(f"data/augmented/{lang}/entity_sub.conll")
    aug_syn = Path(f"data/augmented/{lang}/synonym_swap.conll")
    aug_comb = Path(f"data/augmented/{lang}/combined_augmented.conll")
    
    print(f"\n--- {lang.upper()} ---")
    print(f"  Gold Train:       {count_conll_sentences(gold_train):5d}")
    print(f"  Gold Dev:         {count_conll_sentences(gold_dev):5d}")
    print(f"  Gold Test:        {count_conll_sentences(gold_test):5d}")
    print(f"  Silver Train:     {count_conll_sentences(silver):5d}")
    print(f"  Pseudo Round 1:   {count_conll_sentences(pseudo_r1):5d}")
    print(f"  Pseudo Round 2:   {count_conll_sentences(pseudo_r2):5d}")
    print(f"  Aug Entity Sub:   {count_conll_sentences(aug_entity):5d}")
    print(f"  Aug Synonym:      {count_conll_sentences(aug_syn):5d}")
    print(f"  Aug Combined:     {count_conll_sentences(aug_comb):5d}")
