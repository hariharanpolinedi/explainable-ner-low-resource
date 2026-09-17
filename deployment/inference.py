"""
Phase 15: Production Inference Module for Low-Resource Multilingual NER.

Provides an optimized, in-memory cached inference engine for Konkani, Assamese, and Sinhala:
- Raw text input -> Phase-2 text normalization (normalize.py)
- Script-aware word tokenization (tokenizer.py)
- Padded tensor collation and forward pass through frozen Phase-12 final model
- CRF Viterbi decoding into BIO tags
- Strict entity span extraction with surface strings and character offsets
- Phase-13 Gradient × Attention attribution computation for predicted entities
- In-memory caching so models are loaded only once per process
- GPU-with-CPU-fallback device management
"""

from __future__ import annotations

import html
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding for non-ASCII scripts on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn

from evaluation.seqeval_metrics import extract_entities
from explainability.attention_extractor import AttentionExtractor
from explainability.gradient_attention_attribution import GradientAttentionAttributor
from model.ner_model_v4_multilingual import NERMultilingualModel
from preprocessing.normalize import normalize_text
from preprocessing.tokenizer import ScriptTokenizer
from training.multilingual_dataset import LANG_TO_ID
from training.train_charcnn import ID_TO_TAG, TAG_TO_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ("konkani", "assamese", "sinhala")
DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "deployment" / "bundles"
DEFAULT_BUNDLE_VERSION = "v1"

# In-memory predictor cache: (language, bundle_version, device_str) -> NERPredictor
_PREDICTOR_CACHE: dict[tuple[str, str, str], NERPredictor] = {}


def select_device(preferred_device: str | torch.device | None = None) -> torch.device:
    """
    Select execution device with graceful CPU fallback.
    """
    if preferred_device is not None:
        if isinstance(preferred_device, str):
            device = torch.device(preferred_device)
        else:
            device = preferred_device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


class NERPredictor:
    """
    High-performance, standalone predictor for a single language using bundled artifacts.
    """

    def __init__(
        self,
        language: str,
        bundle_dir: Path | str | None = None,
        bundle_version: str = DEFAULT_BUNDLE_VERSION,
        device: torch.device | str | None = None,
    ) -> None:
        self.language = language.lower().strip()
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. Supported languages: {list(SUPPORTED_LANGUAGES)}"
            )

        self.bundle_version = bundle_version
        if bundle_dir is None:
            self.bundle_path = DEFAULT_BUNDLE_DIR / self.language / bundle_version
        else:
            self.bundle_path = Path(bundle_dir) / self.language / bundle_version

        if not self.bundle_path.is_dir():
            raise FileNotFoundError(
                f"Deployment bundle not found at {self.bundle_path}. Run bundle_artifacts.py first."
            )

        self.device = select_device(device)
        device_name = (
            torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        )
        logger.info(
            "Initializing NERPredictor for '%s' on device %s (%s)",
            self.language,
            self.device,
            device_name,
        )

        # 1. Load Vocabularies
        w_vocab_path = self.bundle_path / "word_vocab.json"
        c_vocab_path = self.bundle_path / "char_vocab.json"
        with open(w_vocab_path, "r", encoding="utf-8") as f:
            w_data = json.load(f)
            self.word_to_id: dict[str, int] = w_data.get("token_to_id", w_data)
        with open(c_vocab_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
            self.char_to_id: dict[str, int] = c_data.get("token_to_id", c_data)

        # 2. Load Model Checkpoint
        model_path = self.bundle_path / "model.pt"
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)

        self.model = NERMultilingualModel(
            word_vocab_size=len(self.word_to_id),
            char_vocab_size=len(self.char_to_id),
            num_tags=9,
            num_languages=3,
            hidden_dim=128,
        ).to(self.device)

        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

        # 3. Setup Preprocessing and Explainability
        self.tokenizer = ScriptTokenizer(self.language)
        self.lang_id = LANG_TO_ID[self.language]

        self.extractor = AttentionExtractor(
            model=self.model,
            word_to_id=self.word_to_id,
            char_to_id=self.char_to_id,
            lang=self.language,
            lang_id=self.lang_id,
            device=self.device,
        )
        self.attributor = GradientAttentionAttributor(self.extractor)

    def predict(
        self,
        sentence: str,
        include_attribution: bool = True,
        normalize_input: bool = True,
    ) -> dict[str, Any]:
        """
        Run end-to-end inference and attribution on a raw sentence.
        """
        if not sentence or not sentence.strip():
            return {
                "language": self.language,
                "raw_sentence": sentence,
                "normalized_sentence": "",
                "tokens": [],
                "tags": [],
                "entities": [],
                "device": str(self.device),
            }

        # 1. Normalize text (Phase 2 canonical Unicode NFC)
        norm_text = normalize_text(sentence) if normalize_input else sentence.strip()

        # 2. Script-aware tokenization
        tokens = self.tokenizer.tokenize_words(norm_text)
        if not tokens:
            return {
                "language": self.language,
                "raw_sentence": sentence,
                "normalized_sentence": norm_text,
                "tokens": [],
                "tags": [],
                "entities": [],
                "device": str(self.device),
            }

        # 3. Model Forward Pass & Viterbi Decoding
        w, c, l, mask = self.extractor.prepare_input_tensors(tokens)
        with torch.no_grad():
            paths = self.model(w, c, lang_ids=l, mask=mask)

        pred_tags = [ID_TO_TAG.get(tid, "O") for tid in paths[0][: len(tokens)]]

        # 4. Entity Span Extraction & Character Offset Alignment
        raw_spans = extract_entities(pred_tags)  # set of (etype, start_tok, end_tok) [end exclusive]
        sorted_spans = sorted(raw_spans, key=lambda x: (x[1], x[2]))

        entities: list[dict[str, Any]] = []
        token_char_spans = self._compute_token_char_offsets(norm_text, tokens)

        for etype, s_idx, e_idx in sorted_spans:
            ent_tokens = tokens[s_idx:e_idx]
            surface_text = " ".join(ent_tokens)

            # Character boundary offsets
            start_char = token_char_spans[s_idx][0] if s_idx < len(token_char_spans) else -1
            end_char = token_char_spans[e_idx - 1][1] if e_idx - 1 < len(token_char_spans) else -1
            if start_char != -1 and end_char != -1 and end_char >= start_char:
                surface_text = norm_text[start_char:end_char]

            ent_record: dict[str, Any] = {
                "type": etype,
                "text": surface_text,
                "start_token": s_idx,
                "end_token": e_idx,
                "start_char": start_char,
                "end_char": end_char,
            }

            # 5. Compute Attribution for predicted entity
            if include_attribution:
                target_token_idx = s_idx
                target_tag_id = TAG_TO_ID.get(f"B-{etype}", 0)
                try:
                    attr_result = self.attributor.attribute_token(
                        tokens=tokens,
                        target_token_idx=target_token_idx,
                        target_tag_id=target_tag_id,
                        normalize=True,
                    )
                    attr_scores = attr_result["attribution_scores"].detach().cpu().tolist()
                    ent_record["attributions"] = [round(score, 4) for score in attr_scores]
                    ent_record["top_attended_tokens"] = [
                        {"token": t, "score": round(sc, 4), "index": idx}
                        for t, sc, idx in attr_result["ranked_tokens"][:5]
                    ]
                except Exception as e:
                    logger.warning("Attribution calculation failed for entity '%s': %s", surface_text, e)
                    ent_record["attributions"] = [0.0] * len(tokens)
                    ent_record["top_attended_tokens"] = []

            entities.append(ent_record)

        return {
            "language": self.language,
            "raw_sentence": sentence,
            "normalized_sentence": norm_text,
            "tokens": tokens,
            "tags": pred_tags,
            "entities": entities,
            "device": str(self.device),
        }

    def _compute_token_char_offsets(self, text: str, tokens: list[str]) -> list[tuple[int, int]]:
        """
        Find character start/end offsets of each token in the normalized text.
        """
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for tok in tokens:
            pos = text.find(tok, cursor)
            if pos != -1:
                offsets.append((pos, pos + len(tok)))
                cursor = pos + len(tok)
            else:
                offsets.append((cursor, cursor + len(tok)))
                cursor += len(tok)
        return offsets


def get_predictor(
    language: str,
    bundle_dir: Path | str | None = None,
    bundle_version: str = DEFAULT_BUNDLE_VERSION,
    device: torch.device | str | None = None,
) -> NERPredictor:
    """
    Retrieve or instantiate a cached NERPredictor for the requested language and device.
    """
    lang = language.lower().strip()
    dev = select_device(device)
    key = (lang, bundle_version, str(dev))

    if key not in _PREDICTOR_CACHE:
        logger.info("Cache miss for predictor %s. Loading bundle...", key)
        _PREDICTOR_CACHE[key] = NERPredictor(
            language=lang,
            bundle_dir=bundle_dir,
            bundle_version=bundle_version,
            device=dev,
        )
    return _PREDICTOR_CACHE[key]


def clear_predictor_cache() -> None:
    """Clear in-memory predictor cache to free memory."""
    global _PREDICTOR_CACHE
    _PREDICTOR_CACHE.clear()


if __name__ == "__main__":
    test_cases = [
        ("konkani", "ताचो जल्म नावेली साश्ट गोंय हांगा जालो ."),
        ("assamese", "১৯২১ চনত মোহনদাস গান্ধী ভাৰতীয় কংগ্ৰেছ দলত যোগদান কৰে ।"),
        ("sinhala", "ශ්‍රී ලංකා ජනාධිපති කොළඹ නගරයේ සංචාරය කළේය ."),
    ]

    for lang, sent in test_cases:
        print(f"\nTesting Inference for {lang.upper()}:")
        predictor = get_predictor(lang)
        result = predictor.predict(sent, include_attribution=True)
        print(f"  Sentence: {result['normalized_sentence']}")
        print(f"  Tokens: {result['tokens']}")
        print(f"  Tags:   {result['tags']}")
        print(f"  Entities Found: {len(result['entities'])}")
        for e in result["entities"]:
            print(f"    - {e['text']} ({e['type']}) [tokens {e['start_token']}:{e['end_token']}]")
            if "top_attended_tokens" in e:
                top3 = [f"{item['token']} ({item['score']})" for item in e["top_attended_tokens"][:3]]
                print(f"      Top Attributions: {', '.join(top3)}")
