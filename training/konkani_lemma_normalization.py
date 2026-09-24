"""
Konkani Lemma/Stem Normalization at Input Layer & Retraining (Cycle 2).

Directly addresses the confirmed root cause from Step 1:
- Strips known case-marker suffixes (locative -ांत/-ंत, genitive -च्या/-चो, dative -क/-ाक,
  ergative -ांनी/-ान/-ेन, and irregular stems like गोयाच्या -> गोंय, पणजींत -> पणजी)
  before word-embedding lookup.
- If a token is OOV in word_vocab.json but its stripped base form is in-vocabulary,
  the word-embedding lookup receives the base form's representation.
- The Char-CNN continues to receive the exact surface graphemes, preserving subword
  morphological distinction for the encoder and CRF.
- Evaluated as an isolated candidate and retrained from checkpoints/improved_konkani_v2.pt.
"""

from __future__ import annotations

import json
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
from torch.utils.data import DataLoader, Dataset

from evaluation.phase5_targeted_interventions import (
    evaluate_model_metrics,
    load_or_create_multilingual_vocabs,
)
from model.ner_model_v4_multilingual import NERMultilingualModel
from preprocessing.tokenizer import ScriptTokenizer
from training.checkpoint_divergence_gate import save_patched_checkpoint_with_gate
from training.multilingual_dataset import LANG_TO_ID
from training.train_charcnn import ID_TO_TAG, TAG_TO_ID
from utils.seed import set_global_seed
from weak_supervision.estimate_silver_precision import load_conll_data

# Irregular stem dictionary for Konkani entities
KONKANI_STEM_MAP: dict[str, str] = {
    "गोयाच्या": "गोंय",
    "गोयांत": "गोंय",
    "गोयाक": "गोंय",
    "गोयान": "गोंय",
    "गोया": "गोंय",
    "पणजींत": "पणजी",
    "पणजीक": "पणजी",
    "पणजीन": "पणजी",
    "मडगांवांत": "मडगांव",
    "मडगांवाक": "मडगांव",
    "मडगांवाच्या": "मडगांव",
    "वास्कोक": "वास्को",
    "वास्कोच्या": "वास्को",
    "म्हापश्यांत": "म्हापसा",
    "म्हापश्याच्या": "म्हापसा",
    "म्हापशाक": "म्हापसा",
    "कुडचड्यांत": "कुडचडें",
    "कुडचड्याक": "कुडचडें",
    "भारतांत": "भारत",
    "भारताच्या": "भारत",
    "भारताक": "भारत",
    "भारतान": "भारत",
    "भारतीयांक": "भारतीय",
    "दिल्लींत": "दिल्ली",
    "दिल्लीक": "दिल्ली",
    "मुंबयंत": "मुंबय",
    "मुंबयक": "मुंबय",
    "सावंतांनी": "सावंत",
    "सावंतांच्या": "सावंत",
    "नायकांनी": "नाईक",
    "नायकांच्या": "नाईक",
    "कामातींनी": "कामत",
    "कामातीच्या": "कामत",
    "आंबेडकरांनी": "आंबेडकर",
    "पर्रीकारांनी": "पर्रीकार",
    "मावज्यांनी": "मावजो",
}

# Explicit suffix stripping rules
KONKANI_STRIP_RULES = [
    # Locative
    (re.compile(r"(.+)ांत$"), r"\1"),
    (re.compile(r"(.+)ंत$"), r"\1"),
    (re.compile(r"(.+)चेर$"), r"\1"),
    (re.compile(r"(.+)कडे$"), r"\1"),
    # Genitive
    (re.compile(r"(.+)च्या$"), r"\1"),
    (re.compile(r"(.+)ाच्या$"), r"\1"),
    (re.compile(r"(.+)चो$"), r"\1"),
    (re.compile(r"(.+)ाचो$"), r"\1"),
    (re.compile(r"(.+)ची$"), r"\1"),
    (re.compile(r"(.+)ाची$"), r"\1"),
    (re.compile(r"(.+)चे$"), r"\1"),
    (re.compile(r"(.+)ाचे$"), r"\1"),
    # Dative / Accusative
    (re.compile(r"(.+)ाक$"), r"\1"),
    (re.compile(r"(.+)क$"), r"\1"),
    (re.compile(r"(.+)का$"), r"\1"),
    # Ergative / Instrumental
    (re.compile(r"(.+)ांनी$"), r"\1"),
    (re.compile(r"(.+)ंनी$"), r"\1"),
    (re.compile(r"(.+)ान$"), r"\1"),
    (re.compile(r"(.+)ेन$"), r"\1"),
    (re.compile(r"(.+)च्यान$"), r"\1"),
]


def normalize_konkani_word_id(
    word: str,
    word_to_id: dict[str, int],
) -> int:
    """
    Look up word_id with lemma normalization:
    1. Direct match in vocabulary -> return ID.
    2. Irregular stem lookup -> return base ID if in-vocab.
    3. Suffix stripping -> return stripped base ID if in-vocab.
    4. Fallback -> return <UNK> ID.
    """
    if word in word_to_id:
        return word_to_id[word]

    # Check irregular map
    if word in KONKANI_STEM_MAP:
        base = KONKANI_STEM_MAP[word]
        if base in word_to_id:
            return word_to_id[base]

    # Check suffix stripping rules
    for pattern, repl in KONKANI_STRIP_RULES:
        m = pattern.match(word)
        if m and len(m.group(1)) >= 2:
            base = pattern.sub(repl, word)
            if base in word_to_id:
                return word_to_id[base]

    return word_to_id.get("<UNK>", 1)


class KonkaniLemmaNormalizedDataset(Dataset):
    """
    Dataset that applies lemma normalization to word_ids while preserving
    exact surface graphemes for char_ids.
    """

    def __init__(
        self,
        conll_paths: list[Path | str],
        word_to_id: dict[str, int],
        char_to_id: dict[str, int],
        apply_normalization: bool = True,
    ):
        self.word_to_id = word_to_id
        self.char_to_id = char_to_id
        self.apply_normalization = apply_normalization
        self.tokenizer = ScriptTokenizer("konkani")
        self.sentences: list[tuple[list[str], list[str]]] = []

        for p in conll_paths:
            sents = load_conll_data(p)
            self.sentences.extend(sents)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        toks, tags = self.sentences[idx]
        word_ids = []
        char_ids_list = []
        tag_ids = [TAG_TO_ID.get(t, 0) for t in tags]

        for tok in toks:
            if self.apply_normalization:
                w_id = normalize_konkani_word_id(tok, self.word_to_id)
            else:
                w_id = self.word_to_id.get(tok, self.word_to_id.get("<UNK>", 1))
            word_ids.append(w_id)

            # Characters always use the exact surface token
            graphemes = self.tokenizer.tokenize_characters(tok)
            c_ids = [self.char_to_id.get(g, self.char_to_id.get("<UNK>", 1)) for g in graphemes]
            if not c_ids:
                c_ids = [1]
            char_ids_list.append(c_ids)

        return {
            "tokens": toks,
            "word_ids": word_ids,
            "char_ids": char_ids_list,
            "tag_ids": tag_ids,
            "lang": "konkani",
        }


def collate_fn_lemma_norm(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_seq_len = max(len(item["word_ids"]) for item in batch)
    max_word_len = max(
        max(len(c_list) for c_list in item["char_ids"]) if item["char_ids"] else 1
        for item in batch
    )

    word_ids_t = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    char_ids_t = torch.zeros((batch_size, max_seq_len, max_word_len), dtype=torch.long)
    tags_t = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    mask_t = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    lang_ids_t = torch.full((batch_size,), LANG_TO_ID["konkani"], dtype=torch.long)

    for b_idx, item in enumerate(batch):
        slen = len(item["word_ids"])
        word_ids_t[b_idx, :slen] = torch.tensor(item["word_ids"], dtype=torch.long)
        tags_t[b_idx, :slen] = torch.tensor(item["tag_ids"], dtype=torch.long)
        mask_t[b_idx, :slen] = True

        for w_idx, c_list in enumerate(item["char_ids"]):
            char_ids_t[b_idx, w_idx, : len(c_list)] = torch.tensor(c_list, dtype=torch.long)

    return {
        "word_ids": word_ids_t,
        "char_ids": char_ids_t,
        "tags": tags_t,
        "mask": mask_t,
        "lang_ids": lang_ids_t,
    }


def evaluate_with_lemma_normalization(
    model: nn.Module,
    conll_path: Path | str,
    word_to_id: dict[str, int],
    char_to_id: dict[str, int],
    device: torch.device,
    apply_normalization: bool = True,
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    Evaluates model with or without lemma normalization on any CoNLL file.
    """
    ds = KonkaniLemmaNormalizedDataset(
        conll_paths=[conll_path],
        word_to_id=word_to_id,
        char_to_id=char_to_id,
        apply_normalization=apply_normalization,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_lemma_norm)

    model.eval()
    all_gold_tags: list[list[str]] = []
    all_pred_tags: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            w_ids = batch["word_ids"].to(device)
            c_ids = batch["char_ids"].to(device)
            l_ids = batch["lang_ids"].to(device)
            mask = batch["mask"].to(device)

            paths = model(w_ids, c_ids, lang_ids=l_ids, mask=mask)

            for i in range(len(paths)):
                slen = mask[i].sum().item()
                gold_seq = [ID_TO_TAG[tid] for tid in batch["tags"][i][:slen].tolist()]
                pred_seq = [ID_TO_TAG[tid] for tid in paths[i][:slen]]
                all_gold_tags.append(gold_seq)
                all_pred_tags.append(pred_seq)

    from evaluation.seqeval_metrics import compute_seqeval_metrics
    from weak_supervision.estimate_silver_precision import extract_entities_from_bio

    metrics = compute_seqeval_metrics(all_gold_tags, all_pred_tags)

    num_sentences = len(all_gold_tags)
    collapse_count = 0
    sents_with_entities = 0

    for s_idx in range(num_sentences):
        g_seq = all_gold_tags[s_idx]
        p_seq = all_pred_tags[s_idx]
        g_spans = list(extract_entities_from_bio(g_seq))
        p_spans = list(extract_entities_from_bio(p_seq))

        if len(g_spans) > 0:
            sents_with_entities += 1
            if len(p_spans) == 0:
                collapse_count += 1

    collapse_rate = (collapse_count / sents_with_entities * 100) if sents_with_entities > 0 else 0.0

    return {
        "entity_f1": round(metrics["overall"]["entity_f1"] * 100, 2),
        "entity_precision": round(metrics["overall"]["entity_precision"] * 100, 2),
        "entity_recall": round(metrics["overall"]["entity_recall"] * 100, 2),
        "token_accuracy": round(metrics["overall"]["token_accuracy"] * 100, 2),
        "by_type": {
            k: {
                "f1": round(v["f1"] * 100, 2),
                "precision": round(v["precision"] * 100, 2),
                "recall": round(v["recall"] * 100, 2),
            }
            for k, v in metrics.get("by_type", {}).items()
        },
        "collapse_count": collapse_count,
        "sents_with_entities": sents_with_entities,
        "collapse_rate": round(collapse_rate, 2),
    }


def predict_sentence_lemma_norm(
    model: nn.Module,
    tokens: list[str],
    word_to_id: dict[str, int],
    char_to_id: dict[str, int],
    device: torch.device,
    apply_normalization: bool = True,
) -> list[str]:
    tokenizer = ScriptTokenizer("konkani")
    max_len = len(tokens)
    if max_len == 0:
        return []

    w_batch = torch.zeros((1, max_len), dtype=torch.long)
    mask = torch.ones((1, max_len), dtype=torch.bool)
    tok_graphemes = [tokenizer.tokenize_characters(w) for w in tokens]
    max_c = max(len(gs) for gs in tok_graphemes) if tok_graphemes else 1
    c_batch = torch.zeros((1, max_len, max_c), dtype=torch.long)
    l_batch = torch.tensor([LANG_TO_ID["konkani"]], dtype=torch.long)

    for j, (w, gs) in enumerate(zip(tokens, tok_graphemes)):
        if apply_normalization:
            w_id = normalize_konkani_word_id(w, word_to_id)
        else:
            w_id = word_to_id.get(w, word_to_id.get("<UNK>", 1))
        w_batch[0, j] = w_id
        for k, g in enumerate(gs):
            c_batch[0, j, k] = char_to_id.get(g, char_to_id.get("<UNK>", 1))

    with torch.no_grad():
        paths = model(
            w_batch.to(device),
            c_batch.to(device),
            lang_ids=l_batch.to(device),
            mask=mask.to(device),
        )
    return [ID_TO_TAG[int(tid)] for tid in paths[0][:max_len]]


def train_konkani_normalization_candidate(
    baseline_ckpt_path: Path,
    output_ckpt_path: Path,
    epochs: int = 5,
    lr: float = 5e-5,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Retrains Konkani model from checkpoints/improved_konkani_v2.pt
    on Gold Train (2x oversampled) with input-layer lemma normalization active.
    """
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    gold_dir = PROJECT_ROOT / "data" / "gold" / "konkani"

    print("\n" + "=" * 80)
    print(f"KONKANI CANDIDATE 2: LEMMA NORMALIZATION RETRAINING (Device: {device})")
    print(f"Starting Baseline Checkpoint: {baseline_ckpt_path.name}")
    print("=" * 80)

    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)

    train_paths = [
        gold_dir / "train.conll",
        gold_dir / "train.conll",
    ]

    train_ds = KonkaniLemmaNormalizedDataset(
        conll_paths=train_paths,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
        apply_normalization=True,
    )
    dev_path = gold_dir / "dev.conll"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_lemma_norm,
    )

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

    ckpt = torch.load(baseline_ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)

    baseline_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate before retraining with normalization active
    pre_eval = evaluate_with_lemma_normalization(
        model, dev_path, word_to_id, char_to_id, device, apply_normalization=True
    )
    print(f"Pre-retrain (Zero-shot Lemma Norm) Dev F1: {pre_eval['entity_f1']:.2f}% (P: {pre_eval['entity_precision']:.2f}%, R: {pre_eval['entity_recall']:.2f}%)")

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
        ep_eval = evaluate_with_lemma_normalization(
            model, dev_path, word_to_id, char_to_id, device, apply_normalization=True
        )
        cur_f1 = ep_eval["entity_f1"]
        print(f"  Epoch {ep:2d}/{epochs:2d} - Loss: {avg_loss:.4f} | Dev F1: {cur_f1:.2f}% (P: {ep_eval['entity_precision']:.2f}%, R: {ep_eval['entity_recall']:.2f}%)")

        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_eval = ep_eval
            best_cand_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_cand_state is not None

    metadata = {
        "candidate": "cycle2_konkani_normalization",
        "epochs_trained": epochs,
        "best_dev_f1": best_f1,
        "dev_eval": best_eval,
        "seed": seed,
    }

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
        "candidate": "cycle2_konkani_normalization",
        "best_dev_f1": best_f1,
        "dev_eval": best_eval,
        "divergence": div,
        "checkpoint_path": str(output_ckpt_path),
    }


def train_konkani_combined_candidate(
    baseline_ckpt_path: Path,
    output_ckpt_path: Path,
    aug_conll_path: Path,
    epochs: int = 5,
    lr: float = 5e-5,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Retrains Konkani model combining both Candidate 1 (Augmentation) and
    Candidate 2 (Lemma Normalization).
    """
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    gold_dir = PROJECT_ROOT / "data" / "gold" / "konkani"

    print("\n" + "=" * 80)
    print(f"KONKANI CANDIDATE 3: COMBINED (AUGMENTATION + LEMMA NORMALIZATION) (Device: {device})")
    print(f"Starting Baseline Checkpoint: {baseline_ckpt_path.name}")
    print("=" * 80)

    word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)

    train_paths = [
        gold_dir / "train.conll",
        gold_dir / "train.conll",
        aug_conll_path,
    ]

    train_ds = KonkaniLemmaNormalizedDataset(
        conll_paths=train_paths,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
        apply_normalization=True,
    )
    dev_path = gold_dir / "dev.conll"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_lemma_norm,
    )

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

    ckpt = torch.load(baseline_ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)

    baseline_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

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
        ep_eval = evaluate_with_lemma_normalization(
            model, dev_path, word_to_id, char_to_id, device, apply_normalization=True
        )
        cur_f1 = ep_eval["entity_f1"]
        print(f"  Epoch {ep:2d}/{epochs:2d} - Loss: {avg_loss:.4f} | Dev F1: {cur_f1:.2f}% (P: {ep_eval['entity_precision']:.2f}%, R: {ep_eval['entity_recall']:.2f}%)")

        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_eval = ep_eval
            best_cand_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_cand_state is not None

    metadata = {
        "candidate": "cycle2_konkani_combined",
        "epochs_trained": epochs,
        "best_dev_f1": best_f1,
        "dev_eval": best_eval,
        "seed": seed,
    }

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
        "candidate": "cycle2_konkani_combined",
        "best_dev_f1": best_f1,
        "dev_eval": best_eval,
        "divergence": div,
        "checkpoint_path": str(output_ckpt_path),
    }


if __name__ == "__main__":
    base_ckpt = PROJECT_ROOT / "checkpoints" / "improved_konkani_v2.pt"
    out_ckpt_norm = PROJECT_ROOT / "checkpoints" / "cycle2_konkani_normalization.pt"
    train_konkani_normalization_candidate(base_ckpt, out_ckpt_norm, epochs=5, lr=5e-5)

    aug_path = PROJECT_ROOT / "data" / "augmented" / "konkani" / "inflectional_cycle2.conll"
    out_ckpt_comb = PROJECT_ROOT / "checkpoints" / "cycle2_konkani_combined.pt"
    train_konkani_combined_candidate(base_ckpt, out_ckpt_comb, aug_path, epochs=5, lr=5e-5)

