"""
Phase 10/11 Patch: Dedicated Institutional ORG Rule Tagger.

Implements deep morphological and syntactic pattern matching for Organizations
in Konkani, Assamese, and Sinhala:
1. Multi-token institutional head matching (ministries, councils, boards, corporations,
   universities, commissions, trusts, media bodies, banks, political parties)
2. Case-inflected suffixes (genitive, locative, dative inflections on institutional terms)
3. Geographic and administrative modifier span-expansion (e.g. "State X Department", "National Y Council")
4. Conjunction-aware compound organization names ("X and Y Medical College", "A and B Hospital")
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PUNCTUATION_TOKENS = {
    ".", ",", "!", "?", "।", "॥", ";", ":", "-", "—", "(", ")",
    "[", "]", "{", "}", "\"", "'", "«", "»", "෴", "%", "/", "@", "&"
}

# Institutional Head Keywords (Base and compound forms)
ORG_HEAD_KEYWORDS: dict[str, set[str]] = {
    "konkani": {
        "मंत्रालय", "सचिवालय", "खातें", "मंडळ", "महामंडळ", "आयोग", "परिशद", "परिषद",
        "समिती", "संस्था", "विद्यापीठ", "महाविद्यालय", "न्यायालय", "बँक", "फौंडेशन",
        "ट्रस्ट", "सोसायटी", "अकादमी", "अकादेमी", "पक्ष", "संघटना", "कार्यालय",
        "रुग्णालय", "केंद्र", "संघ", "युनियन", "बोर्ड", "कंपनी", "उद्योग", "प्राधिकरण",
        "असोसिएशन", "क्लब", "महापालिका", "नगरपालिका", "पंचायत", "सेना", "दल", "मोर्चा"
    },
    "assamese": {
        "মন্ত্রণালয়", "সঞ্চালকালয়", "বিভাগ", "নিগম", "পৰিষদ", "সমিতি", "সংস্থা",
        "আয়োগ", "ন্যায়ালয়", "আদালত", "বিশ্ববিদ্যালয়", "মহাবিদ্যালয়", "প্ৰতিষ্ঠান",
        "সংসদ", "কেন্দ্ৰ", "মণ্ডল", "সংঘ", "সন্থা", "সমাজ", "দল", "বেংক", "কোম্পানী",
        "হাস্পাতাল", "চিকিৎসালয়", "ইউনিয়ন", "ব’ৰ্ড", "প্ৰাধিকৰণ", "সমবায়", "প্ৰেছ",
        "মহাসভা", "সভা", "মঞ্চ", "মৰ্চা", "আৰক্ষী", "ফৌজ", "বাহিনী", "সেনা"
    },
    "sinhala": {
        "අමාත්‍යාංශය", "දෙපාර්තමේන්තුව", "මණ්ඩලය", "සංස්ථාව", "කොමිසම", "අධිකාරිය",
        "ආයතනය", "විශ්වවිද්‍යාලය", "විද්‍යාලය", "පීඨය", "අධිකරණය", "මහාධිකරණය",
        "බැංකුව", "පක්ෂය", "පෙරමුණ", "සංගමය", "සමිතිය", "රෝහල", "සභාව", "මධ්‍යස්ථානය",
        "කාර්යාලය", "කමිටුව", "සමාගම", "පදනම", "ව්‍යාපාරය", "සේවය", "හමුදාව", "පොලීසිය",
        "සංවිධානය", "ආරක්ෂක", "සභා", "කොට්ඨාසය"
    },
}

# Modifiers that commonly precede institutional heads (State, Central, National, High, etc.)
ORG_PREFIX_MODIFIERS: dict[str, set[str]] = {
    "konkani": {
        "गोवा", "गोंय", "राज्य", "केंद्रीय", "राष्ट्रीय", "भारतीय", "अखिल", "जिल्हा",
        "सरकारी", "सार्वजनिक", "उच्च", "सर्वोच्च", "लोक", "विकास", "महिला", "बाल",
        "शिक्षण", "आरोग्य", "सहकारी", "तंत्रज्ञान", "वैद्यकीय", "अभियांत्रिकी"
    },
    "assamese": {
        "অসম", "ৰাজ্যিক", "কেন্দ্ৰীয়", "ৰাষ্ট্ৰীয়", "ভাৰতীয়", "সদৌ", "জিলা",
        "চৰকাৰী", "ৰাজহুৱা", "উচ্চ", "উচ্চতম", "লোক", "উন্নয়ন", "মহিলা", "শিশু",
        "শিক্ষা", "স্বাস্থ্য", "সমবায়", "প্ৰযুক্তি", "চিকিৎসা", "অভিযান্ত্ৰিক", "কৃষি"
    },
    "sinhala": {
        "ශ්‍රී", "ලංකා", "ජාතික", "මධ්‍යම", "පළාත්", "රාජ්‍ය", "මහ", "උසස්", "ශ්‍රේෂ්ඨ",
        "දිස්ත්‍රික්", "මහා", "පොදු", "සමූපකාර", "විවෘත", "තාක්ෂණ", "වෛද්‍ය", "ඉංජිනේරු"
    },
}

# Internal Conjunctions allowed inside compound organization names
ORG_INTERNAL_CONJUNCTIONS: dict[str, set[str]] = {
    "konkani": {"आनी", "आणि", "व"},
    "assamese": {"আৰু", "বা", "ও"},
    "sinhala": {"හා", "සහ"},
}


class DedicatedORGRuleTagger:
    """
    High-precision institutional organization tagger with morphological expansion.
    """

    def __init__(self, lang: str) -> None:
        self.lang = lang.lower().strip()
        self.head_keywords = ORG_HEAD_KEYWORDS.get(self.lang, set())
        self.prefix_modifiers = ORG_PREFIX_MODIFIERS.get(self.lang, set())
        self.conjunctions = ORG_INTERNAL_CONJUNCTIONS.get(self.lang, set())

    def _matches_head(self, token: str) -> bool:
        """Check if a token matches or ends with an institutional head keyword."""
        if token in self.head_keywords:
            return True
        for kw in self.head_keywords:
            if len(kw) >= 3 and token.endswith(kw):
                return True
        return False

    def tag_sentence(self, tokens: Sequence[str]) -> list[str]:
        """
        Tag sentence tokens with B-ORG / I-ORG using morphological & syntactic rules.
        """
        n = len(tokens)
        tags = ["O"] * n
        if n == 0:
            return tags

        i = 0
        while i < n:
            tok = tokens[i]
            if tok in PUNCTUATION_TOKENS:
                i += 1
                continue

            # Check if current token is an institutional head keyword
            if self._matches_head(tok):
                # Expand span leftwards to capture modifiers and named entities
                start_idx = i
                while start_idx > 0:
                    prev_tok = tokens[start_idx - 1]
                    if prev_tok in PUNCTUATION_TOKENS:
                        break
                    # If preceding token is a known prefix modifier, proper noun, or conjunct
                    if prev_tok in self.prefix_modifiers:
                        start_idx -= 1
                    elif start_idx >= 2 and tokens[start_idx - 1] in self.conjunctions:
                        # e.g., "Medical and Health Department"
                        start_idx -= 2
                    elif len(prev_tok) >= 3 and prev_tok not in {"ह्या", "तो", "ती", "ते", "हांगा", "जालो", "আছিল", "হল", "විය", "ඇත"}:
                        start_idx -= 1
                        # Limit maximum leftward expansion to 4 tokens
                        if i - start_idx >= 4:
                            break
                    else:
                        break

                # Expand span rightwards if followed by conjunction + secondary head
                end_idx = i + 1
                if end_idx + 1 < n and tokens[end_idx] in self.conjunctions:
                    next_tok = tokens[end_idx + 1]
                    if self._matches_head(next_tok) or next_tok in self.head_keywords:
                        end_idx = end_idx + 2

                # Apply tags
                if end_idx > start_idx:
                    tags[start_idx] = "B-ORG"
                    for j in range(start_idx + 1, end_idx):
                        tags[j] = "I-ORG"
                    i = end_idx
                    continue

            i += 1

        # Strict invariant: Punctuation is never ORG
        for k in range(n):
            if tokens[k] in PUNCTUATION_TOKENS:
                tags[k] = "O"

        # BIO integrity fix
        for k in range(n):
            if tags[k] == "I-ORG":
                if k == 0 or tags[k - 1] not in ("B-ORG", "I-ORG"):
                    tags[k] = "B-ORG"

        return tags
