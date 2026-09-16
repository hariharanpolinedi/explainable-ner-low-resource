"""
From-Scratch Byte-Pair Encoding (BPE) Tokenizer for Low-Resource Languages.

Ablation module for Phase 12 (to compare subword vs word+grapheme modeling).
Strictly trains from scratch on this project's own Phase 1 corpus;
zero external tokenizer models, checkpoints, or libraries are imported.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import BaseConfig, load_config, parse_cli_args
from utils.logger import ExperimentLogger
from utils.seed import set_global_seed


class FromScratchBPETokenizer:
    """
    Pure Python Byte-Pair Encoding tokenizer fitted strictly on local corpora.
    """

    def __init__(self, vocab_size: int = 8000) -> None:
        self.target_vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}

    def get_stats(self, vocab_counts: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
        """Count frequencies of adjacent symbol pairs."""
        pairs: Counter[tuple[str, str]] = Counter()
        for word, freq in vocab_counts.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def merge_vocab(
        self,
        pair: tuple[str, str],
        vocab_counts: dict[tuple[str, ...], int],
    ) -> dict[tuple[str, ...], int]:
        """Merge all occurrences of a specific pair across vocabulary words."""
        new_vocab: dict[tuple[str, ...], int] = {}
        bigram = pair
        replacement = pair[0] + pair[1]

        for word, freq in vocab_counts.items():
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def fit(self, sentences: list[str], max_merges: int = 1000) -> None:
        """
        Fit BPE merges and vocabulary on a list of whitespace-tokenized sentences.
        """
        # Count word frequencies and represent words as character tuples ending in '</w>'
        word_freqs: Counter[str] = Counter()
        for sent in sentences:
            for token in sent.split():
                if token:
                    word_freqs[token] += 1

        # Initial vocabulary: characters plus end-of-word token '</w>'
        vocab_counts: dict[tuple[str, ...], int] = {
            tuple(list(word) + ["</w>"]): freq for word, freq in word_freqs.items()
        }

        # Initialize base character vocabulary
        initial_chars: set[str] = set()
        for word in vocab_counts.keys():
            initial_chars.update(word)

        print(f"Initial base character vocabulary size: {len(initial_chars)}")
        self.merges = []

        # Learn merges iteratively
        for i in range(max_merges):
            pairs = self.get_stats(vocab_counts)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)  # type: ignore
            if pairs[best_pair] < 2:  # Stop if pair frequency drops below 2
                break

            vocab_counts = self.merge_vocab(best_pair, vocab_counts)
            self.merges.append(best_pair)

        # Build final vocabulary map
        final_vocab_tokens: set[str] = set(["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
        for word in vocab_counts.keys():
            final_vocab_tokens.update(word)

        self.vocab = {tok: idx for idx, tok in enumerate(sorted(final_vocab_tokens))}
        self.inverse_vocab = {idx: tok for tok, idx in self.vocab.items()}
        print(f"Trained BPE with {len(self.merges)} merges. Final vocabulary size: {len(self.vocab)}")

    def encode_word(self, word: str) -> list[str]:
        """Encode a single word into learned BPE subword units."""
        pieces = tuple(list(word) + ["</w>"])
        for bigram in self.merges:
            if len(pieces) <= 1:
                break
            replacement = bigram[0] + bigram[1]
            new_pieces: list[str] = []
            i = 0
            while i < len(pieces):
                if i < len(pieces) - 1 and (pieces[i], pieces[i + 1]) == bigram:
                    new_pieces.append(replacement)
                    i += 2
                else:
                    new_pieces.append(pieces[i])
                    i += 1
            pieces = tuple(new_pieces)
        return list(pieces)

    def encode(self, sentence: str) -> list[str]:
        """Encode a sentence into a sequence of BPE subwords."""
        tokens: list[str] = []
        for word in sentence.split():
            tokens.extend(self.encode_word(word))
        return tokens

    def decode(self, subwords: list[str]) -> str:
        """Decode a sequence of BPE subwords back to normalized text."""
        text = "".join(subwords)
        return text.replace("</w>", " ").strip()

    def save(self, output_dir: Path, lang: str) -> None:
        """Save BPE merges and vocabulary."""
        output_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = output_dir / f"bpe_vocab_{lang}.json"
        merges_path = output_dir / f"bpe_merges_{lang}.txt"

        with open(vocab_path, "w", encoding="utf-8") as fp:
            json.dump(self.vocab, fp, ensure_ascii=False, indent=2)

        with open(merges_path, "w", encoding="utf-8") as fp:
            for pair in self.merges:
                fp.write(f"{pair[0]} {pair[1]}\n")

        print(f"Saved BPE vocabulary ({len(self.vocab)}) and {len(self.merges)} merges to {output_dir}")


def run_bpe_training(config_path: str = "configs/base_config.yaml", max_merges: int = 500) -> dict[str, Any]:
    cfg: BaseConfig = load_config(config_path)
    set_global_seed(cfg.seed)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    results: dict[str, Any] = {}

    for lang in cfg.project.languages:
        sent_path = processed_dir / lang / "sentences.txt"
        if not sent_path.is_file():
            continue

        # Load first 10,000 sentences for fast ablation BPE fitting
        sample_sentences: list[str] = []
        with open(sent_path, "r", encoding="utf-8", errors="ignore") as fp:
            for i, line in enumerate(fp):
                if i >= 10000:
                    break
                sample_sentences.append(line.strip())

        print(f"[{lang.upper()}] Fitting from-scratch BPE on {len(sample_sentences)} sample sentences...")
        bpe = FromScratchBPETokenizer(vocab_size=8000)
        bpe.fit(sample_sentences, max_merges=max_merges)
        bpe.save(processed_dir / lang, lang)

        # Test encoding on sample
        test_sent = sample_sentences[0] if sample_sentences else ""
        encoded = bpe.encode(test_sent)
        decoded = bpe.decode(encoded)

        results[f"{lang}_bpe_vocab_size"] = len(bpe.vocab)
        results[f"{lang}_bpe_merges"] = len(bpe.merges)
        safe_preview = [ascii(x) for x in encoded[:5]]
        print(f"[{lang.upper()}] Test encode ({len(encoded)} subwords): {safe_preview}...")

    return results


if __name__ == "__main__":
    cfg_file, _ = parse_cli_args(sys.argv[1:])
    run_bpe_training(cfg_file, max_merges=500)
