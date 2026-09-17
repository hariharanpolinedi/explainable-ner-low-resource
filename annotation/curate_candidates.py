"""
Curate 35 sentences per language with diverse named entities from the raw pool.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation.select_long_multientity_v2 import load_used_sentences, count_list_delimiters

# Distinctive entity anchor keywords to find balanced sentences
ANCHORS = {
    "konkani": {
        "PER": ["सावंत", "नाईक", "कामत", "गांधी", "नेहरू", "आंबेडकर", "मावजो", "बोरकार", "सरदेसाय", "प्रमोद", "मनोहर", "दामोदर"],
        "LOC": ["पणजी", "गोवा", "गोंय", "मडगांव", "वास्को", "म्हापसा", "फोंडा", "दिल्ली", "मुंबई", "भारत", "महाराष्ट्र", "कर्नाटक"],
        "ORG": ["सरकार", "विधानसभा", "विद्यापीठ", "मंडळ", "समिती", "परिषद", "बँक", "काँग्रेस", "भाजपा", "संस्था", "आयोग"],
        "MISC": ["पुरस्कार", "शिगमो", "चवथ", "संविधान", "कायदा", "उत्सव", "दिवाळी", "अकादेमी"],
    },
    "assamese": {
        "PER": ["শৰ্মা", "গোস্বামী", "বৰুৱা", "হাজৰিকা", "গান্ধী", "নেহৰু", "হিমন্ত", "ভূপেন", "শঙ্কৰদেৱ", "বৰদলৈ"],
        "LOC": ["গুৱাহাটী", "অসম", "যোৰহাট", "ডিব্ৰুগড়", "শিলচৰ", "তেজপুৰ", "নগাঁও", "ব্ৰহ্মপুত্ৰ", "ভাৰত", "দিল্লী", "বৰপেটা"],
        "ORG": ["চৰকাৰ", "বিধানসভা", "বিশ্ববিদ্যালয়", "সমিতি", "সভা", "পৰিষদ", "দল", "সংস্থা", "আদালত", "বেংক"],
        "MISC": ["বিহু", "বঁটা", "সংবিধান", "আইন", "উৎসৱ", "চুক্তি", "সাহিত্য"],
    },
    "sinhala": {
        "PER": ["රාජපක්ෂ", "වික්‍රමසිංහ", "දිසානායක", "ප්‍රේමදාස", "සේනානායක", "බණ්ඩාරනායක", "මහින්ද", "රනිල්", "අනුර"],
        "LOC": ["කොළඹ", "නුවර", "ගාල්ල", "යාපනය", "මාතර", "අනුරාධපුරය", "ශ්‍රී ලංකා", "ලංකාව", "ඉන්දියාව"],
        "ORG": ["රජය", "පාර්ලිමේන්තුව", "විශ්වවිද්‍යාලය", "සමිතිය", "පක්ෂය", "මණ්ඩලය", "සංස්ථාව", "හමුදාව", "පොලීසිය"],
        "MISC": ["අවුරුද්ද", "වෙසක්", "පොසොන්", "පෙරහැර", "සම්මානය", "ව්‍යවස්ථාව", "පනත"],
    },
}


def find_balanced_candidates(lang: str, count: int = 35) -> list[dict]:
    used = load_used_sentences(lang)
    raw_p = PROJECT_ROOT / "data" / "processed" / lang / "sentences.txt"
    anchors = ANCHORS[lang]

    selected = []
    seen = set()

    with open(raw_p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s in used or s in seen:
                continue
            toks = s.split()
            if not (30 <= len(toks) <= 80):
                continue
            if count_list_delimiters(toks, lang) < 2:
                continue

            # Check if sentence has anchors from any category
            has_per = any(any(a in t for a in anchors["PER"]) for t in toks)
            has_loc = any(any(a in t for a in anchors["LOC"]) for t in toks)
            has_org = any(any(a in t for a in anchors["ORG"]) for t in toks)
            has_misc = any(any(a in t for a in anchors["MISC"]) for t in toks)

            # We want sentences containing at least one named entity cue
            if has_per or has_loc or has_org or has_misc:
                seen.add(s)
                selected.append({
                    "id": f"{lang}_v2_{len(selected)+1:03d}",
                    "tokens": toks,
                    "text": s,
                    "cues": {"PER": has_per, "LOC": has_loc, "ORG": has_org, "MISC": has_misc},
                })
                if len(selected) >= count:
                    break

    return selected


if __name__ == "__main__":
    for lang in ["konkani", "assamese", "sinhala"]:
        res = find_balanced_candidates(lang, 35)
        print(f"[{lang.upper()}] Found {len(res)} balanced candidates.")
        cues_summary = Counter()
        for r in res:
            for k, v in r["cues"].items():
                if v:
                    cues_summary[k] += 1
        print(f"  Cue distribution: {dict(cues_summary)}")
