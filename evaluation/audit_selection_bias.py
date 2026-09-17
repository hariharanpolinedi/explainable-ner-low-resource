"""
Audit Selection Bias in Long Multi-Entity Gold Evaluation Set Creation.

Traces the candidate-selection logic in annotation/expand_long_multientity_set.py:
1. Filters applied to raw sentences.
2. Ranking / screening criteria.
3. Quantifies how ORG keywords, gazetteers, and rule taggers influenced selection.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation import expand_long_multientity_set
from weak_supervision.org_rule_taggers import DedicatedORGRuleTagger, ORG_HEAD_KEYWORDS, ORG_PREFIX_MODIFIERS


def audit_candidate_selection_logic() -> dict[str, Any]:
    """Inspect the source code of expand_long_multientity_set.py directly."""
    src = inspect.getsource(expand_long_multientity_set.select_and_annotate_expansion_sentences)

    has_screening_sets = "has_gaz = any(t in gaz_first_words for t in toks)" in src
    has_org_screen = "has_org = any(t in org_head_keywords or t in org_modifiers for t in toks)" in src
    has_combined_screen = "if not (has_gaz or has_org):" in src

    # Check rule tagger injection
    has_rule_tagger_injection = "org_tags = org_tagger.tag_sentence(toks)" in src
    has_adjudicated_overwrite = 'if adjudicated_tags[i] == "O" and org_tags[i] != "O":' in src

    # Check stopping condition and candidate acceptance
    has_acceptance_filter = 'orgs = [e for e in ents if e[0] == "ORG"]' in src
    has_mid_or_org_req = "if len(orgs) > 0 or len(mids) > 0:" in src
    has_target_break = "if total_org_count >= target_orgs and total_mid_count >= target_middle and len(selected_sentences) >= 30:" in src

    findings = {
        "screening_filters": {
            "min_token_len": 28,
            "max_token_len": 85,
            "has_gaz_or_org_screen": has_combined_screen,
            "has_gaz_screen": has_screening_sets,
            "has_org_screen": has_org_screen,
        },
        "rule_tagger_injected_into_gold": {
            "present": has_rule_tagger_injection and has_adjudicated_overwrite,
            "source_code_snippet": """
org_tags = org_tagger.tag_sentence(toks)
adjudicated_tags = list(tags_a)
for i in range(len(toks)):
    if adjudicated_tags[i] == "O" and org_tags[i] != "O":
        adjudicated_tags[i] = org_tags[i]
""".strip()
        },
        "acceptance_criteria": {
            "has_acceptance_filter": has_acceptance_filter,
            "has_mid_or_org_req": has_mid_or_org_req,
            "has_target_break": has_target_break,
        },
    }

    return findings


def simulate_bias_effect_on_raw_corpus(lang: str, sample_limit: int = 2000) -> dict[str, Any]:
    """Sample raw 30+ token sentences with and without ORG/gazetteer screening to quantify natural vs selected distribution."""
    raw_file = PROJECT_ROOT / "data" / "processed" / lang / "sentences.txt"
    org_tagger = DedicatedORGRuleTagger(lang)
    org_heads = set(org_tagger.head_keywords)
    org_mods = set(org_tagger.prefix_modifiers)

    total_long = 0
    long_with_org_cue = 0

    if not raw_file.exists():
        return {"error": f"{raw_file} not found"}

    with open(raw_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens = line.strip().split()
            if 28 <= len(tokens) <= 85:
                total_long += 1
                if any(t in org_heads or t in org_mods for t in tokens):
                    long_with_org_cue += 1
                if total_long >= sample_limit:
                    break

    pct_cued = (long_with_org_cue / total_long * 100) if total_long > 0 else 0.0
    return {
        "total_sampled_long_sentences": total_long,
        "long_sentences_with_org_cue": long_with_org_cue,
        "percentage_with_org_cue": round(pct_cued, 2),
    }


if __name__ == "__main__":
    findings = audit_candidate_selection_logic()
    print("=== SELECTION BIAS AUDIT FINDINGS ===")
    import pprint
    pprint.pprint(findings)

    print("\n=== RAW CORPUS SCREENING PREVALENCE ===")
    for lang in ["konkani", "assamese", "sinhala"]:
        stats = simulate_bias_effect_on_raw_corpus(lang)
        print(f"[{lang.upper()}]: {stats}")
