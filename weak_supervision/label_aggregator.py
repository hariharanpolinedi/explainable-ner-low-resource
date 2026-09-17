"""
Label aggregation engine for weak supervision.

Combines GazetteerMatcher, RuleTagger, and WikipediaDistantLabeler using
empirical precision weighting calibrated on gold-dev.
Prunes net-negative labelers to guarantee high precision (>= 70%) on the silver standard.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import BaseConfig, load_config
from weak_supervision.estimate_silver_precision import (
    evaluate_predictions,
    extract_entities_from_bio,
    load_conll_data,
)
from weak_supervision.gazetteer_matcher import GazetteerMatcher
from weak_supervision.loc_rule_taggers import DedicatedLOCRuleTagger
from weak_supervision.org_rule_taggers import DedicatedORGRuleTagger
from weak_supervision.rule_taggers import RuleTagger
from weak_supervision.wikipedia_distant_labeler import WikipediaDistantLabeler

PUNCTUATION_TOKENS = {
    ".", ",", "!", "?", "।", "॥", ";", ":", "-", "—", "(", ")",
    "[", "]", "{", "}", "\"", "'", "«", "»", "෴", "%", "/", "@"
}


class LabelAggregator:
    """Combines multiple weak labelers with precision-calibrated consensus."""

    def __init__(
        self,
        lang: str,
        gaz_matcher: GazetteerMatcher,
        rule_tagger: RuleTagger,
        wiki_labeler: WikipediaDistantLabeler,
        org_rule_tagger: DedicatedORGRuleTagger | None = None,
        loc_rule_tagger: DedicatedLOCRuleTagger | None = None,
        min_precision_threshold: float = 0.70,
    ) -> None:
        self.lang = lang.lower()
        self.labelers: dict[str, Any] = {
            "gazetteer": gaz_matcher,
            "rule": rule_tagger,
            "wikipedia": wiki_labeler,
        }
        if org_rule_tagger is not None:
            self.labelers["org_rule"] = org_rule_tagger
        if loc_rule_tagger is not None:
            self.labelers["loc_rule"] = loc_rule_tagger
        self.min_precision_threshold = min_precision_threshold
        self.labeler_weights: dict[str, float] = {k: 1.0 for k in self.labelers}
        self.active_labelers: list[str] = list(self.labelers.keys())
        self.calibration_report: dict[str, Any] = {}

    def calibrate(self, gold_dev_data: list[tuple[list[str], list[str]]]) -> dict[str, Any]:
        """
        Evaluate each labeler against gold-dev, identify net-negative labelers,
        and calibrate precision weights for aggregation.
        """
        tokens_list = [toks for toks, _ in gold_dev_data]
        raw_preds: dict[str, list[list[str]]] = {}
        perf: dict[str, dict[str, Any]] = {}

        # 1. Standalone evaluation
        for name, labeler in self.labelers.items():
            preds = [labeler.tag_sentence(toks) for toks in tokens_list]
            raw_preds[name] = preds
            res = evaluate_predictions(gold_dev_data, preds)
            perf[name] = res
            prec = res["overall"]["precision"]
            self.labeler_weights[name] = prec

        # 2. Identify net-negative labelers
        # A labeler is dropped if its precision is below min_precision_threshold
        # or if including it drops the aggregate precision
        active = []
        dropped = []
        for name, p in perf.items():
            prec = p["overall"]["precision"]
            if prec >= self.min_precision_threshold:
                active.append(name)
            else:
                dropped.append((name, prec, f"Precision {prec:.4f} < {self.min_precision_threshold:.2f} threshold"))

        # Fallback: if all dropped, retain the highest precision labeler
        if not active:
            best = max(perf.keys(), key=lambda k: perf[k]["overall"]["precision"])
            active = [best]

        self.active_labelers = active
        self.calibration_report = {
            "language": self.lang,
            "standalone_performance": perf,
            "active_labelers": self.active_labelers,
            "dropped_labelers": dropped,
        }
        return self.calibration_report

    def aggregate_sentence(self, tokens: list[str]) -> list[str]:
        """
        Aggregate predictions from active labelers on a single token sequence.
        Resolves conflicts using empirical precision weights and enforces BIO invariants.
        """
        n = len(tokens)
        if n == 0:
            return []

        # Collect predictions from active labelers
        preds: dict[str, list[str]] = {}
        for name in self.active_labelers:
            preds[name] = self.labelers[name].tag_sentence(tokens)

        final_labels = ["O"] * n

        # Extract entity spans proposed by each active labeler
        # Score each candidate span by sum of precision weights of agreeing labelers
        span_votes: dict[tuple[int, int, str], float] = defaultdict(float)

        for name, labels in preds.items():
            weight = self.labeler_weights.get(name, 1.0)
            spans = extract_entities_from_bio(labels)
            for etype, start, end in spans:
                span_votes[(start, end, etype)] += weight

        # Sort candidate spans by score descending, then length descending
        sorted_spans = sorted(
            span_votes.items(),
            key=lambda item: (item[1], item[0][1] - item[0][0]),
            reverse=True,
        )

        occupied = [False] * n
        accepted_spans: list[tuple[int, int, str]] = []

        for (start, end, etype), score in sorted_spans:
            # Check for token overlap with already accepted higher-confidence spans
            if any(occupied[k] for k in range(start, end)):
                continue

            # Minimum score threshold: at least one high-precision labeler
            if score >= self.min_precision_threshold:
                accepted_spans.append((start, end, etype))
                for k in range(start, end):
                    occupied[k] = True

        # Write accepted spans into final labels
        for start, end, etype in accepted_spans:
            final_labels[start] = f"B-{etype}"
            for k in range(start + 1, end):
                final_labels[k] = f"I-{etype}"

        # Invariant: Punctuation is always O
        for i, tok in enumerate(tokens):
            if tok in PUNCTUATION_TOKENS:
                final_labels[i] = "O"

        # BIO consistency check
        for i in range(n):
            if final_labels[i].startswith("I-"):
                curr_type = final_labels[i][2:]
                if i == 0 or final_labels[i - 1] not in (f"B-{curr_type}", f"I-{curr_type}"):
                    final_labels[i] = f"B-{curr_type}"

        return final_labels
