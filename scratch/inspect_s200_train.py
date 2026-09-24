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

s200_toks, s200_gold = load_conll_data(PROJECT_ROOT / "data/gold/assamese/long_multientity_eval_v2.conll")[0]
spans = list(extract_entities_from_bio(s200_gold))
print("Sentence 200 Gold Entities:")
for etype, s, e in spans:
    ent_str = " ".join(s200_toks[s:e])
    tok_pairs = [(s200_toks[i], s200_gold[i]) for i in range(s, e)]
    print(f"  [{s:2d}:{e:2d}] {ent_str:30s} -> {etype:5s} | {tok_pairs}")

train_orig = load_conll_data(PROJECT_ROOT / "data/gold/assamese/train.conll")
train_corr = load_conll_data(PROJECT_ROOT / "data/gold/assamese/train_cycle2_corrected.conll")

target_tokens = ["ভাৰতৰ", "ভাৰতীয়", "ৰিজাৰ্ভ", "বেংক", "ষ্টেট", "জীৱন", "বিমা", "নিগম", "মুম্বাইত"]
print("\nToken occurrences in train.conll vs train_cycle2_corrected.conll:")
for tok in target_tokens:
    orig_tags = [t for s in train_orig for w, t in zip(s[0], s[1]) if w == tok]
    corr_tags = [t for s in train_corr for w, t in zip(s[0], s[1]) if w == tok]
    o_orig = orig_tags.count("O")
    ent_orig = len(orig_tags) - o_orig
    o_corr = corr_tags.count("O")
    ent_corr = len(corr_tags) - o_corr
    print(f"  {tok:12s}: Orig Total={len(orig_tags):2d} (O={o_orig:2d}, Ent={ent_orig:2d}) | Corr Total={len(corr_tags):2d} (O={o_corr:2d}, Ent={ent_corr:2d})")

# Let's inspect where ভাৰতৰ, ভাৰতীয়, মুম্বাইত, আদি appear in raw/silver data
silver_path = PROJECT_ROOT / "data/silver/assamese/train.conll"
if silver_path.exists():
    silver_sents = load_conll_data(silver_path)
    print(f"\nSilver Assamese data: {len(silver_sents)} sentences.")
    for tok in ["ভাৰতৰ", "মুম্বাইত", "ৰিজাৰ্ভ বেংক"]:
        count = sum(1 for s in silver_sents if any(tok in " ".join(s[0]) for _ in [1]))
        print(f"  Silver occurrences of '{tok}': {count} sentences")
