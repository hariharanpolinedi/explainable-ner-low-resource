"""
Multilingual Dataset and Batch Sampler Pipeline (Phase 8).

Handles:
1. Building/loading unified multilingual word and character vocabularies across Konkani, Assamese, and Sinhala.
2. Data pooling of gold-train + silver data tagged with language IDs.
3. Batch sampling strategies: Uniform sampling (equal probability per language) and Proportional sampling (weighted by corpus volume).
4. Dynamic sequence and character tensor collation.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch.utils.data import Dataset, Sampler
except ImportError:
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    Sampler = object  # type: ignore

from preprocessing.tokenizer import ScriptTokenizer
from training.train_charcnn import TAG_TO_ID

LANG_TO_ID = {
    "konkani": 0,
    "assamese": 1,
    "sinhala": 2,
}
ID_TO_LANG = {v: k for k, v in LANG_TO_ID.items()}

# Raw corpus sentence volumes from Phase 1 (reports/corpus_stats.md)
# Used for proportional sampling
RAW_CORPUS_VOLUMES = {
    "konkani": 30980,
    "assamese": 55002,
    "sinhala": 66954,
}


def load_or_create_multilingual_vocabs(
    proc_dir: Path | str,
    languages: Sequence[str] = ("konkani", "assamese", "sinhala"),
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Load or generate unified word and character vocabularies across all specified languages.
    Preserves special tokens: <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3.
    """
    proc_dir = Path(proc_dir)
    multi_dir = proc_dir / "multilingual"
    multi_dir.mkdir(parents=True, exist_ok=True)

    word_vocab_path = multi_dir / "word_vocab.json"
    char_vocab_path = multi_dir / "char_vocab.json"

    if word_vocab_path.is_file() and char_vocab_path.is_file():
        with open(word_vocab_path, "r", encoding="utf-8") as fp:
            w_data = json.load(fp)
            word_to_id = w_data["token_to_id"]
        with open(char_vocab_path, "r", encoding="utf-8") as fp:
            c_data = json.load(fp)
            char_to_id = c_data["token_to_id"]
        return word_to_id, char_to_id

    # Build unified word vocab
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    unique_words: set[str] = set()
    unique_chars: set[str] = set()

    for lang in languages:
        l_w_path = proc_dir / lang / "word_vocab.json"
        l_c_path = proc_dir / lang / "char_vocab.json"
        if l_w_path.is_file():
            with open(l_w_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                for tok in data.get("token_to_id", {}).keys():
                    if tok not in special_tokens:
                        unique_words.add(tok)
        if l_c_path.is_file():
            with open(l_c_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                for tok in data.get("token_to_id", {}).keys():
                    if tok not in special_tokens:
                        unique_chars.add(tok)

    # Sort deterministically
    sorted_words = sorted(list(unique_words))
    sorted_chars = sorted(list(unique_chars))

    word_to_id: dict[str, int] = {tok: idx for idx, tok in enumerate(special_tokens)}
    for tok in sorted_words:
        word_to_id[tok] = len(word_to_id)

    char_to_id: dict[str, int] = {tok: idx for idx, tok in enumerate(special_tokens)}
    for tok in sorted_chars:
        char_to_id[tok] = len(char_to_id)

    # Save
    with open(word_vocab_path, "w", encoding="utf-8") as fp:
        json.dump({
            "special_tokens": special_tokens,
            "vocab_size": len(word_to_id),
            "languages": list(languages),
            "token_to_id": word_to_id,
        }, fp, ensure_ascii=False, indent=2)

    with open(char_vocab_path, "w", encoding="utf-8") as fp:
        json.dump({
            "special_tokens": special_tokens,
            "vocab_size": len(char_to_id),
            "languages": list(languages),
            "token_to_id": char_to_id,
        }, fp, ensure_ascii=False, indent=2)

    print(f"Built unified multilingual vocabs: {len(word_to_id)} words, {len(char_to_id)} chars.")
    return word_to_id, char_to_id


class MultilingualNERDataset(Dataset):
    """
    Pooled Multilingual NER dataset with language tagging and character decomposition.
    """

    def __init__(
        self,
        lang_conll_paths: dict[str, list[Path | str]],
        word_to_id: dict[str, int],
        char_to_id: dict[str, int],
        pad_word_id: int = 0,
        unk_word_id: int = 1,
        pad_char_id: int = 0,
        unk_char_id: int = 1,
        max_sentences_per_lang: int | None = None,
    ) -> None:
        self.word_to_id = word_to_id
        self.char_to_id = char_to_id
        self.pad_word_id = pad_word_id
        self.unk_word_id = unk_word_id
        self.pad_char_id = pad_char_id
        self.unk_char_id = unk_char_id
        self.max_sentences_per_lang = max_sentences_per_lang

        self.tokenizers = {
            lang: ScriptTokenizer(lang) for lang in lang_conll_paths.keys()
        }

        # samples: (tokens, word_ids, char_ids_per_word, tag_ids, lang_id)
        self.samples: list[tuple[list[str], list[int], list[list[int]], list[int], int]] = []
        # lang_indices: lang_id -> list of indices in self.samples
        self.lang_indices: dict[int, list[int]] = {
            LANG_TO_ID[lang]: [] for lang in lang_conll_paths.keys()
        }

        self._load_pooled_data(lang_conll_paths)

    def _load_pooled_data(self, lang_conll_paths: dict[str, list[Path | str]]) -> None:
        for lang, paths in lang_conll_paths.items():
            lang_id = LANG_TO_ID[lang]
            tok = self.tokenizers[lang]

            for p in paths:
                if self.max_sentences_per_lang and len(self.lang_indices[lang_id]) >= self.max_sentences_per_lang:
                    break
                fpath = Path(p)
                if not fpath.is_file():
                    raise FileNotFoundError(f"Missing CoNLL file for {lang}: {fpath}")

                curr_tokens: list[str] = []
                curr_tags: list[int] = []

                with open(fpath, "r", encoding="utf-8") as fp:
                    for line in fp:
                        line_str = line.strip()
                        if not line_str:
                            if curr_tokens:
                                self._append_sample(curr_tokens, curr_tags, lang_id, tok)
                                curr_tokens = []
                                curr_tags = []
                                if self.max_sentences_per_lang and len(self.lang_indices[lang_id]) >= self.max_sentences_per_lang:
                                    break
                            continue

                        parts = line_str.split("\t")
                        if len(parts) == 2:
                            tok_str, tag_str = parts
                            curr_tokens.append(tok_str)
                            curr_tags.append(TAG_TO_ID.get(tag_str, 0))

                if curr_tokens and (not self.max_sentences_per_lang or len(self.lang_indices[lang_id]) < self.max_sentences_per_lang):
                    self._append_sample(curr_tokens, curr_tags, lang_id, tok)


    def _append_sample(
        self,
        tokens: list[str],
        tags: list[int],
        lang_id: int,
        tokenizer: ScriptTokenizer,
    ) -> None:
        word_ids = [self.word_to_id.get(t, self.unk_word_id) for t in tokens]
        char_ids_per_word: list[list[int]] = []
        for t in tokens:
            graphemes = tokenizer.tokenize_characters(t)
            c_ids = [self.char_to_id.get(g, self.unk_char_id) for g in graphemes]
            if not c_ids:
                c_ids = [self.unk_char_id]
            char_ids_per_word.append(c_ids)

        idx = len(self.samples)
        self.samples.append((tokens, word_ids, char_ids_per_word, tags, lang_id))
        self.lang_indices[lang_id].append(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[str], list[int], list[list[int]], list[int], int]:
        return self.samples[idx]


class MultilingualBatchSampler(Sampler):
    """
    Batch sampler supporting Uniform and Proportional sampling strategies across languages.
    
    Generates batches where each batch contains samples from a single selected language,
    ensuring efficient padding and direct CRF routing, with the distribution of batches
    governed strictly by `strategy`.
    """

    def __init__(
        self,
        lang_indices: dict[int, list[int]],
        batch_size: int = 32,
        strategy: str = "uniform",
        seed: int = 42,
        num_batches_per_epoch: int | None = None,
    ) -> None:
        self.lang_indices = {k: list(v) for k, v in lang_indices.items()}
        self.batch_size = batch_size
        self.strategy = strategy
        self.seed = seed

        languages = list(self.lang_indices.keys())
        self.languages = languages

        # Compute sampling weights per language
        if strategy == "uniform":
            self.weights = [1.0 / len(languages)] * len(languages)
        elif strategy == "proportional":
            raw_vols = [RAW_CORPUS_VOLUMES[ID_TO_LANG[lid]] for lid in languages]
            tot = sum(raw_vols)
            self.weights = [v / tot for v in raw_vols]
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        # Total batches per epoch: defaults to total samples / batch_size
        total_samples = sum(len(idx_list) for idx_list in self.lang_indices.values())
        if num_batches_per_epoch is None:
            self.num_batches = math.ceil(total_samples / batch_size)
        else:
            self.num_batches = num_batches_per_epoch

        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        # Shuffle per-language pools at epoch start
        shuffled_pools = {
            lid: list(self.rng.sample(indices, len(indices)))
            for lid, indices in self.lang_indices.items()
        }
        pointers = {lid: 0 for lid in self.languages}

        for _ in range(self.num_batches):
            # Select language according to strategy weights
            chosen_lang = self.rng.choices(self.languages, weights=self.weights, k=1)[0]
            pool = shuffled_pools[chosen_lang]
            ptr = pointers[chosen_lang]

            batch_indices: list[int] = []
            while len(batch_indices) < self.batch_size:
                if ptr >= len(pool):
                    # Reshuffle if pool exhausted
                    shuffled_pools[chosen_lang] = list(self.rng.sample(self.lang_indices[chosen_lang], len(self.lang_indices[chosen_lang])))
                    pool = shuffled_pools[chosen_lang]
                    ptr = 0
                take = min(self.batch_size - len(batch_indices), len(pool) - ptr)
                batch_indices.extend(pool[ptr : ptr + take])
                ptr += take

            pointers[chosen_lang] = ptr
            yield batch_indices


def collate_fn_multilingual(
    batch: list[tuple[list[str], list[int], list[list[int]], list[int], int]],
    min_word_len: int = 5,
) -> dict[str, Any]:
    """
    Collate variable-length multilingual samples into padded tensors.
    """
    batch_tokens = [item[0] for item in batch]
    batch_word_ids = [item[1] for item in batch]
    batch_char_ids = [item[2] for item in batch]
    batch_tag_ids = [item[3] for item in batch]
    batch_lang_ids = [item[4] for item in batch]

    batch_size = len(batch)
    seq_lengths = [len(seq) for seq in batch_word_ids]
    max_seq_len = max(seq_lengths) if seq_lengths else 1

    max_char_len = min_word_len
    for sent in batch_char_ids:
        for word_chars in sent:
            if len(word_chars) > max_char_len:
                max_char_len = len(word_chars)

    padded_word_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    padded_char_ids = torch.zeros((batch_size, max_seq_len, max_char_len), dtype=torch.long)
    padded_tags = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    lang_ids = torch.tensor(batch_lang_ids, dtype=torch.long)

    for i in range(batch_size):
        slen = len(batch_word_ids[i])
        padded_word_ids[i, :slen] = torch.tensor(batch_word_ids[i], dtype=torch.long)
        padded_tags[i, :slen] = torch.tensor(batch_tag_ids[i], dtype=torch.long)
        mask[i, :slen] = True

        for j in range(slen):
            w_chars = batch_char_ids[i][j]
            clen = len(w_chars)
            padded_char_ids[i, j, :clen] = torch.tensor(w_chars, dtype=torch.long)

    return {
        "tokens": batch_tokens,
        "word_ids": padded_word_ids,
        "char_ids": padded_char_ids,
        "lang_ids": lang_ids,
        "tags": padded_tags,
        "mask": mask,
        "seq_lengths": seq_lengths,
    }
