"""
PER Honorific & Title Rule Tagger Module.

Audits and strengthens rule-based person-name detection across Konkani, Sinhala, and Assamese.
Handles:
- Academic and professional titles (Prof., Dr., Father)
- Political and civic titles (Minister, Chief Minister, President, MP)
- Religious titles (Reverend, Venerable Thero)
- Postposition honorifics (Mr., Mrs., Miss, Deva)

Guarantees strict Phase-3 annotation alignment:
Titles and honorific prefixes are tagged as 'O', while the person's name tokens
immediately following receive 'B-PER' followed by 'I-PER' tags.
"""

from __future__ import annotations

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

PUNCTUATION_TOKENS = {
    ".", ",", "!", "?", "।", "॥", ";", ":", "-", "—", "(", ")",
    "[", "]", "{", "}", "\"", "'", "«", "»", "෴", "%", "/", "@"
}

# Prefix honorifics (single and multi-token)
HONORIFIC_PREFIXES: dict[str, list[list[str]]] = {
    "konkani": [
        ["डॉ."], ["डॉ"], ["श्री."], ["श्री"], ["श्रीमती."], ["श्रीमती"],
        ["प्रा."], ["प्रा"], ["प्राध्यापक"], ["फा."], ["फादर"],
        ["केंद्रीय", "मंत्री"], ["मंत्री"], ["मुख्यमंत्री"], ["उपमुख्यमंत्री"],
        ["राज्यपाल"], ["आमदार"], ["खासदार"], ["न्यायमूर्ती"],
    ],
    "sinhala": [
        ["මහාචාර්ය"], ["ආචාර්ය"], ["ඩොක්ටර්"], ["ගරු"], ["පූජ්‍ය"],
        ["ජනාධිපති"], ["අගමැති"], ["ඇමති"], ["මන්ත්‍රී"],
    ],
    "assamese": [
        ["ড°"], ["ডাঃ"], ["শ্ৰী"], ["শ্ৰীমতী"], ["প্ৰাঃ"], ["প্ৰাধ্যাপক"],
        ["মুখ্যমন্ত্ৰী"], ["প্ৰধানমন্ত্ৰী"], ["মন্ত্ৰී"], ["ৰাজ্যপাল"], ["ন্যায়াধীশ"],
    ],
}

# Suffix / postposition honorific markers (appear immediately after the person name)
HONORIFIC_SUFFIXES: dict[str, set[str]] = {
    "konkani": {"बाब", "बाबू", "बाय", "शेट", "भट"},
    "sinhala": {"මහතා", "මහතාගේ", "මහත්මිය", "මහත්මියගේ", "මෙනෙවිය", "හිමි", "හිමියන්"},
    "assamese": {"মহাশয়", "মহাশয়া", "দেৱ", "দেৱী", "ডাঙৰীয়া"},
}

# Context particles following person subjects
SUBJECT_PARTICLES: dict[str, set[str]] = {
    "konkani": {"हांणी", "हानी", "कडेन", "सांगता"},
    "sinhala": {"විසින්", "සමඟ", "මගින්"},
    "assamese": {"দ্বাৰা", "লগত", "হঁতে"},
}


class PERHonorificRuleTagger:
    """Rule tagger focused specifically on person entity boundaries and honorific contexts."""

    def __init__(self, lang: str) -> None:
        self.lang = lang.lower()
        self.prefixes = HONORIFIC_PREFIXES.get(self.lang, [])
        self.suffixes = HONORIFIC_SUFFIXES.get(self.lang, set())
        self.particles = SUBJECT_PARTICLES.get(self.lang, set())

    def tag_sentence(self, tokens: list[str]) -> list[str]:
        n = len(tokens)
        tags = ["O"] * n

        # 1. Match prefix honorifics (supporting consecutive titles like 'Chief Minister Dr.')
        i = 0
        while i < n:
            matched_prefix_len = 0
            # Keep consuming consecutive prefixes
            while True:
                step_len = 0
                for pref in sorted(self.prefixes, key=lambda x: -len(x)):
                    plen = len(pref)
                    if i + plen <= n and tokens[i:i + plen] == pref:
                        step_len = plen
                        break
                if step_len > 0:
                    for k in range(i, i + step_len):
                        tags[k] = "O"
                    i += step_len
                    matched_prefix_len += step_len
                else:
                    break

            if matched_prefix_len > 0:
                # Check following tokens for Person name (up to 2-3 tokens before punctuation or particle)
                name_start = i
                name_tokens = []
                curr = name_start
                while curr < n and len(name_tokens) < 3:
                    tok = tokens[curr]
                    if tok in PUNCTUATION_TOKENS or tok in self.particles or tok in self.suffixes:
                        break
                    # Avoid non-name stopwords
                    if tok in ("සහ", "හා", "आनी", "আৰু", "වන", "වූ"):
                        break
                    name_tokens.append(curr)
                    curr += 1

                if name_tokens:
                    tags[name_tokens[0]] = "B-PER"
                    for idx in name_tokens[1:]:
                        tags[idx] = "I-PER"

                i = curr
            else:
                i += 1

        # 2. Match suffix honorifics (e.g. රණසිංහ ප්‍රේමදාස මහතාගේ)
        for j in range(n):
            tok = tokens[j]
            if tok in self.suffixes and j > 0:
                # Previous 1 or 2 tokens form person name if currently O
                p1 = j - 1
                if tags[p1] == "O" and tokens[p1] not in PUNCTUATION_TOKENS:
                    if j >= 2 and tags[j - 2] == "O" and tokens[j - 2] not in PUNCTUATION_TOKENS and len(tokens[j - 2]) >= 3:
                        tags[j - 2] = "B-PER"
                        tags[p1] = "I-PER"
                    else:
                        tags[p1] = "B-PER"

        return tags


if __name__ == "__main__":
    test_sentences = {
        "sinhala": "මහාචාර්ය අනුර සිල්වා සහ ආචාර්ය නිමාලි ප්‍රනාන්දු විසින් පර්යේෂණය කරන ලදී .",
        "konkani": "केंद्रीय मंत्री श्रीपाद नाईक हांणी नव्या प्रकल्पांची घोशणा केली .",
        "assamese": "মুখ্যমন্ত্ৰী ড° হিমন্ত বিশ্ব শৰ্মা উপস্থিত আছিল ।",
    }
    for lang, sent in test_sentences.items():
        tagger = PERHonorificRuleTagger(lang)
        toks = sent.split()
        pred_tags = tagger.tag_sentence(toks)
        print(f"\n[{lang.upper()} RULE TAGGER TEST]")
        for t, tag in zip(toks, pred_tags):
            print(f"  {t:<18}: {tag}")
