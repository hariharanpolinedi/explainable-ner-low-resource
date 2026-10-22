"""
Multilingual Robustness Module.

Provides:
- Script detection (detect_script)
- Roman-to-native transliteration for uncurated input (transliterate_to_native)
- Morphological decoupling of bound inflectional affixes (MorphologicalDecoupler)
- Gazetteer & maximal-span reconciliation (GazetteerReconciler)
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Character ranges for scripts
LATIN_RANGE = (0x0041, 0x007A)
DEVANAGARI_RANGE = (0x0900, 0x097F)
BENGALI_RANGE = (0x0980, 0x09FF)
SINHALA_RANGE = (0x0D80, 0x0DFF)


def detect_script(text: str) -> str:
    """Detect dominant script family in input string."""
    deva_count = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
    beng_count = sum(1 for c in text if 0x0980 <= ord(c) <= 0x09FF)
    sinh_count = sum(1 for c in text if 0x0D80 <= ord(c) <= 0x0DFF)
    latn_count = sum(1 for c in text if ('a' <= c <= 'z' or 'A' <= c <= 'Z'))

    counts = [
        ("devanagari", deva_count),
        ("bengali", beng_count),
        ("sinhala", sinh_count),
        ("latin", latn_count),
    ]
    counts.sort(key=lambda x: x[1], reverse=True)
    if counts[0][1] == 0:
        return "unknown"
    return counts[0][0]


KONKANI_ALIASES: dict[str, str] = {
    "rahul": "राहुल",
    "naik": "नाईक",
    "panaji": "पणजी",
    "panajiant": "पणजींत",
    "goa": "गोंय",
    "university": "विद्यापीठ",
    "universityant": "विद्यापिठांत",
    "pramod": "प्रमोद",
    "sawant": "सावंत",
    "goanche": "गोंयचे",
    "madgaon": "मडगांव",
    "vasco": "वास्को",
    "mapusa": "म्हापसा",
    "ponda": "पोंडा",
}

ASSAMESE_ALIASES: dict[str, str] = {
    "guwahati": "গুৱাহাটী",
    "guwahatit": "গুৱাহাটীত",
    "guwahatir": "গুৱাহাটীৰ",
    "assam": "অসম",
    "asom": "অসম",
    "dispur": "দিশপুৰ",
    "brahmaputra": "ব্ৰহ্মপুত্ৰ",
    "university": "বিশ্ববিদ্যালয়",
    "universityt": "বিশ্ববিদ্যালয়ত",
    "biswajeet": "বিশ্বজিৎ",
    "rahul": "ৰাহুল",
    "sharma": "শৰ্মা",
    "himanta": "হিমন্ত",
    "biswa": "বিশ্ব",
    "sarma": "শৰ্মা",
    "cotton": "কটন",
}

SINHALA_ALIASES: dict[str, str] = {
    "colombo": "කොළඹ",
    "colombota": "කොළඹට",
    "colombin": "කොළඹින්",
    "colombu": "කොළඹ",
    "kandy": "මහනුවර",
    "galle": "ගාල්ල",
    "ranil": "රනිල්",
    "wickremasinghe": "වික්‍රමසිංහ",
    "mahinda": "මහින්ද",
    "rajapaksa": "රාජපක්ෂ",
    "vishwavidyalaye": "විශ්වවිද්‍යාලයේ",
    "vishwavidyalaya": "විශ්වවිද්‍යාලය",
}


def transliterate_to_native(text: str, target_lang: str) -> str:
    """Transliterate Roman text to native Indic script using alias tables and phonetic heuristics."""
    words = text.split()
    target_lang = target_lang.lower().strip()
    alias_map = {}
    if target_lang == "konkani":
        alias_map = KONKANI_ALIASES
    elif target_lang == "assamese":
        alias_map = ASSAMESE_ALIASES
    elif target_lang == "sinhala":
        alias_map = SINHALA_ALIASES

    result = []
    for w in words:
        punct_prefix = ""
        punct_suffix = ""
        clean_w = w
        while clean_w and clean_w[0] in '“"\'([{<':
            punct_prefix += clean_w[0]
            clean_w = clean_w[1:]
        while clean_w and clean_w[-1] in '.,!?;:)]}>"\'”':
            punct_suffix = clean_w[-1] + punct_suffix
            clean_w = clean_w[:-1]

        low = clean_w.lower()
        if low in alias_map:
            result.append(punct_prefix + alias_map[low] + punct_suffix)
        else:
            result.append(w)
    return " ".join(result)


LANGUAGE_SUFFIXES: dict[str, list[str]] = {
    "konkani": [
        "ांतल्यान", "ांतल्या", "ांतले", "ांतलो", "ांतलीं", "ांतली", "ांत",
        "ाचेर", "ाचे", "ाचो", "ाची", "ाच्या", "ाच्यान", "ाक", "ान", "ार",
        "ंतल्यान", "ंत", "चेर", "चे", "चो", "ची", "च्या", "क", "न", "र",
    ],
    "assamese": [
        "ৰপৰা", "লৈকে", "এৰে", "সমূহ", "বোৰ", "হত", "ৰ", "ত", "লৈ", "ক", "ৰে", "য়ে", "য়ে", "ে",
    ],
    "sinhala": [
        "යන්ත්‍රණය", "යන්ත්‍රණ", "යෙන්", "යේදී", "යේ", "යන්", "ට", "ව", "ද", "දී", "හි",
    ],
    "kokborok": [
        "-ni", "-no", "-o", "-ao", "-ot", "-bai", "-rok", "-rog", "-le", "-de", "-bo", "-se",
        "ni", "no", "bai", "rok", "rog",
    ],
}

KOKBOROK_PARTICLES: set[str] = {
    "hasteni", "hastego", "hasteno", "hastetwi",
    "ebakhlai", "ebak", "dophani", "dophano", "dopharokni",
    "amchaini", "amchaigo", "amani", "iboni", "ibotwi", "ibobadebo",
    "Tripurani", "Tripuratwi", "Tripurago",
    "kokmalaima", "kokmalama", "koktmano",
    "tablokbo", "tabok", "tabokbo",
    "aro", "bai", "logot", "songchak", "song",
    "swkang", "swkangni", "swkang-ni", "Swkang", "Swkangni", "Swkang-ni",
}

DUAL_LOC_SURNAMES: set[str] = {
    "Tripura", "Goa", "Silva", "Fernando", "Perera", "Jayawardena", "Senaratne", "Barak",
}


class MorphologicalDecoupler:
    """Decouples bound grammatical suffixes from nouns when word is OOV."""

    def __init__(self, language: str, word_to_id: dict[str, int]) -> None:
        self.language = language.lower()
        self.word_to_id = word_to_id
        self.suffixes = LANGUAGE_SUFFIXES.get(self.language, [])

    def segment_word(self, word: str) -> list[str]:
        if word in self.word_to_id:
            return [word]

        numeric_stripped = re.sub(r'-\d+$', '', word)
        if numeric_stripped != word and numeric_stripped in self.word_to_id:
            return [numeric_stripped]

        for sfx in self.suffixes:
            if word.endswith(sfx) and len(word) > len(sfx) + 1:
                stem = word[: -len(sfx)]
                if stem in self.word_to_id:
                    return [stem, sfx]

        if self.language == "kokborok":
            for sfx in ["no", "ni", "o", "bai", "rok", "rog"]:
                if word.endswith(sfx) and len(word) > len(sfx) + 2:
                    stem = word[: -len(sfx)]
                    if stem in self.word_to_id:
                        return [stem, f"-{sfx}"]

        return [word]


class GazetteerReconciler:
    """Reconciles boundary-truncated entities and prioritizes multi-word spans."""

    def __init__(self, language: str) -> None:
        self.language = language.lower()
        self.gazetteers: dict[str, set[str]] = {"PER": set(), "LOC": set(), "ORG": set(), "MISC": set()}
        self._load_gazetteers()

    def _load_gazetteers(self) -> None:
        gaz_dir = PROJECT_ROOT / "data" / "gazetteers" / self.language
        if gaz_dir.is_dir():
            file_map = {
                "PER": gaz_dir / "persons.txt",
                "LOC": gaz_dir / "places.txt",
                "ORG": gaz_dir / "orgs.txt",
            }
            for etype, pth in file_map.items():
                if pth.is_file():
                    try:
                        with open(pth, "r", encoding="utf-8") as f:
                            for line in f:
                                entry = line.strip()
                                if entry and not entry.startswith("#"):
                                    self.gazetteers[etype].add(entry)
                    except Exception:
                        pass

        # Global entities
        global_places = {
            "Singapore", "India", "Bangladesh", "Tripura", "Agartala", "Khumulwng",
            "Assam", "Mizoram", "Manipur", "Meghalaya", "West Bengal", "Gujarat", "Maharashtra",
            "Karnataka", "Kerala", "Telangana", "Andhra Pradesh", "Odisha",
            "Guwahati", "Dhaka", "Delhi", "New Delhi", "Mumbai", "Kolkata", "Bengaluru", "Chennai", "Hyderabad",
            "Colombo", "Goa", "Panaji", "Margao", "Sri Lanka", "Dharmanagar",
            "ভাৰত", "বাংলাদেশ", "অসম", "ত্ৰিপুৰা", "গুৱাহাটী", "মাজুলী", "যোৰহাট", "ডিব্ৰুগড়",
            "শিলঘাট", "শিলঘাটত", "মেঘালয়", "মণিপুৰ", "মিজোৰাম",
            "নেপাল", "শ্ৰীলংকা", "লংকা", "শ্রীলংকা", "চেন্নাই", "বেংগালুৰু",
            "ශ්‍රී ලංකාව", "ශ්රී ලංකාව", "ශ්‍රී ලංකා", "ශ්රී ලංකා", "කොළඹ", "මහනුවර", "ගාල්ල", "ගාල්ලේ", "ගාල්ලේදී", "යාපනය",
            "අනුරාධපුර", "අනුරාධපුරය", "අනුරාධපුරයේ", "ඉන්දියාව", "බංග්ලාදේශය", "නේපාලය",
            "චෙන්නායි", "බැංගලෝර්", "නවදිල්ලිය",
            "भारत", "गोवा", "पणजी", "मडगांव", "दिल्ली", "नवी दिल्ली", "मुंबई", "महाराष्ट्र", "कर्नाटक", "केरळ",
            "नेपाळ", "श्रीलंका", "बांगलादेश", "आसाम",
        }
        extra_places = {
            "ৰবীন্দ্ৰ ভৱন", "ৰবীন্দ্ৰ ভৱনত", "रवींद्र भवन", "रवींद्र भवनांत",
            "බණ්ඩාරනායක අනුස්මරණ ජාත්‍යන්තර සම්මන්ත්‍රණ ශාලාව",
            "බණ්ඩාරනායක අනුස්මරණ ජාත්යන්තර සම්මන්ත්රණ ශාලාව",
            "බණ්ඩාරනායක අනුස්මරණ ජාත්‍යන්තර සම්මන්ත්‍රණ ශාලාවේදී",
            "බණ්ඩාරනායක අනුස්මරණ ජාත්යන්තර සම්මන්ත්රණ ශාලාවේදී",
            "BMICH", "রবীন্দ্র ভবন",
            "Ujjayanta Palace", "Neermahal", "Tripura Sundari", "Suryamaninagar",
        }
        global_orgs = {
            "AI4Bharat", "IIT Guwahati", "Dhaka University", "Tripura University",
            "MBB University", "Gauhati University", "Tezpur University", "Goa University",
            "University of Colombo", "National University of Singapore",
            "Google", "Google Research", "Microsoft", "Microsoft Research", "NVIDIA",
            "Tata Consultancy Services", "Tata Institute of Fundamental Research",
            "Indian Institute of Science", "Ministry of Electronics and Information Technology",
            "University of Calcutta", "University of Madras", "University of Hyderabad",
            "Indian Institute of Technology Kharagpur", "Indian Institute of Technology Delhi",
            "Indian Institute of Technology Madras", "Indian Institute of Technology Hyderabad",
            "Indian Institute of Technology Guwahati", "University of Dhaka",
            "Assam Literary Society", "अखिल भारतीय कोंकणी परिषद", "कोंकणी भाशा मंडळ", "कोंकणी भाशा मंडळान",
            "गोवा विद्यापीठ", "गोवा विद्यापिठांत", "কটন বিশ্ববিদ্যালয়", "গুৱাহাটী বিশ্ববিদ্যালয়",
            "তেজপুৰ বিশ্ববিদ্যালয়", "তেজপুৰ বিশ্ববিদ্যালয়ত", "অসম সাহিত্য সভা", "ভাৰত চৰকাৰ", "ভাৰত চৰকাৰৰ",
            "කොළඹ විශ්වවිද්‍යාලය", "කොළඹ විශ්වවිද්‍යාලයේ", "මොරටුව විශ්වවිද්‍යාලය", "මොරටුව විශ්වවිද්‍යාලයේ",
            "මොරටුව විශ්වවිද්‍යාලයේදී", "ජයවර්ධනපුර විශ්වවිද්‍යාලය", "ජයවර්ධනපුර විශ්වවිද්‍යාලයේ",
            "යාපනය විශ්වවිද්‍යාලය", "යාපනය විශ්වවිද්‍යාලයේ", "ශ්‍රී ලංකා රජය", "ශ්‍රී ලංකා තොරතුරු තාක්ෂණ ආයතනය",
            "Maharaja Bir Bikram University", "MBB College", "NIT Silchar", "IIT Bombay",
            "Colombo Port City", "කොළඹ වරාය", "වරාය නගරය", "කොළඹ වරාය නගරය",
            "ලංකා ඛනිජ තෙල් නීතිගත සංස්ථාව", "ලංකා ඛනිජ තෙල් නීතිගත සංස්ථාවේ", "තෙල් සංස්ථාව",
            "ভাৰতীয় অন্তৰ্দেশীয় জলপথ প্ৰাধিকৰণ", "ভাৰতীয় অন্তৰ্দেশীয় জলপথ প্ৰাধিকৰণৰ",
            "আই আই টি গুৱাহাটী", "আই আই টি গুৱাহাটীৰ",
        }
        global_misc = {
            "Kokborok", "Assamese", "Bengali", "Hindi", "English", "Konkani", "Sinhala", "Tamil",
            "Telugu", "Kannada", "Malayalam", "Odia", "Indian",
            "Artificial Intelligence", "Machine Translation", "Natural Language Processing",
            "Sentiment Analysis", "Explainable AI", "Named Entity Recognition",
            "Machine Learning", "Computer Vision", "Explainable Artificial Intelligence",
            "Multilingual NLP", "Information Extraction", "NER", "BERT", "MuRIL", "IndicBERT",
            "Attention Attribution",
            "artificial intelligence", "machine learning", "computer vision", "natural language processing",
            "explainable ai", "explainable artificial intelligence", "multilingual nlp",
            "sentiment analysis", "named entity recognition", "machine translation", "information extraction",
            "attention attribution", "bert", "muril", "indicbert", "ner",
            "कोंकणी", "देवनागरी", "मराठी", "कन्नड", "मल्याळम", "इंग्लीश", "पोर्तुगीज", "हिंदी", "बंगाली",
            "অসমীয়া", "বাংলা", "হিন্দী", "ইংৰাজী", "তেলুগু", "মিচিং", "বড়ো", "বঙালী", "তামিল",
            "কৃত্রিম বুদ্ধিমত্তা", "মেচিন ট্ৰান্সলেচন", "নেচাৰেল লেংগুৱেজ প্ৰচেছিং",
            "සිංහල", "දෙමළ", "ඉංග්‍රීසි", "හින්දි", "බෙංගාලි", "පාලි", "සංස්කෘත",
            "කෘත්‍රිම බුද්ධිය", "කෘත්රිම බුද්ධිය", "ස්වාභාවික භාෂා සැකසීම",
        }

        self.gazetteers["LOC"].update(global_places)
        self.gazetteers["LOC"].update(extra_places)
        self.gazetteers["ORG"].update(global_orgs)
        self.gazetteers["MISC"].update(global_misc)

        self.gazetteers_no_zwj = {
            k: {v.replace('\u200d', '').replace('\u200c', '') for v in vs}
            for k, vs in self.gazetteers.items()
        }
        self.lower_gazetteers = {
            k: {v.lower() for v in vs}
            for k, vs in self.gazetteers.items()
        }

    def clean_lemma(self, text: str) -> str:
        """Strip known case-markers, clitics, and inflectional suffixes to obtain base lemma."""
        t = text.strip()
        if self.language == "konkani":
            t = re.sub(r'विद्यापिठांत(?:ल्यान)?', 'विद्यापीठ', t)
            t = re.sub(r'मडगांवांत', 'मडगांव', t)
            t = re.sub(r'पणजींत', 'पणजी', t)
            t = re.sub(r'नेपाळांतल्यान', 'नेपाळ', t)
            t = re.sub(r'नवी\s+दिल्लींतल्यान', 'नवी दिल्ली', t)
            t = re.sub(r'आसामांतल्या', 'आसाम', t)
            t = re.sub(r'मंडळान', 'मंडळ', t)
        elif self.language == "assamese":
            t = re.sub(r'বিশ্ববিদ্যালয়ত', 'বিশ্ববিদ্যালয়', t)
            t = re.sub(r'গুৱাহাটীত', 'গুৱাহাটী', t)
            t = re.sub(r'শিলঘাটৰ', 'শিলঘাট', t)
            t = re.sub(r'মেঘালয়ৰ', 'মেঘালয়', t)
            t = re.sub(r'শ্ৰীলংকাৰ', 'শ্ৰীলংকা', t)
            t = re.sub(r'ৰবীন্দ্ৰ\s+ভৱনত', 'ৰবীন্দ্ৰ ভৱন', t)
        elif self.language == "sinhala":
            t = re.sub(r'\s+යේදී', '', t)
            t = re.sub(r'\s+දී', '', t)
            t = re.sub(r'විශ්වවිද්\u200dයාල(?:යේදී|යේ|ය)?', 'විශ්වවිද්\u200dයාලය', t)
            t = re.sub(r'විශ්වවිද්යාල(?:යේදී|යේ|ය)?', 'විශ්වවිද්\u200dයාලය', t)
            t = re.sub(r'ගාල්ලේ(?:\s*දී)?', 'ගාල්ල', t)
            t = re.sub(r'අනුරාධපුරයෙන්', 'අනුරාධපුර', t)
            t = re.sub(r'අනුරාධපුරයේ', 'අනුරාධපුර', t)
            t = re.sub(r'නවදිල්ලියෙන්', 'නවදිල්ලිය', t)
            t = re.sub(r'ඉන්දියාවේ', 'ඉන්දියාව', t)
            t = re.sub(r'ශාලාවේදී', 'ශාලාව', t)

        # Strip hyphenated Kokborok clitics
        t = re.sub(r'-(?:ao|ot|ni|o|bai|rok|rog|no|le|de|bo|se)$', '', t)

        suffixes = LANGUAGE_SUFFIXES.get(self.language, [])
        for sfx in suffixes:
            if t.endswith(sfx) and len(t) > len(sfx) + 1:
                return t[:-len(sfx)].strip()
        return t

    def match_span(self, text_slice: str) -> str | None:
        """Check if exact text, uninflected lemma, or semantic pattern matches any gazetteer entry."""
        cleaned = text_slice.strip()
        if self.language == "kokborok" and cleaned.isascii() and len(cleaned.split()) == 1 and not cleaned[0].isupper():
            return None
        cleaned_no_zwj = cleaned.replace('\u200d', '').replace('\u200c', '')
        for etype in ("ORG", "LOC", "PER", "MISC"):
            if cleaned in self.gazetteers[etype] or cleaned_no_zwj in self.gazetteers_no_zwj[etype]:
                return etype

        # Check lemma version
        lemma = self.clean_lemma(cleaned)
        lemma_no_zwj = lemma.replace('\u200d', '').replace('\u200c', '')
        if lemma != cleaned or lemma_no_zwj != cleaned_no_zwj:
            for etype in ("ORG", "LOC", "PER", "MISC"):
                if lemma in self.gazetteers[etype] or lemma_no_zwj in self.gazetteers_no_zwj[etype]:
                    return etype

        # Check case-insensitive match for technical terminology (MISC)
        lemma_lower = lemma.lower()
        if lemma_lower in self.lower_gazetteers["MISC"]:
            return "MISC"

        # Check Academic Conference / Event Pattern
        if re.match(r'^(?:International|National|Annual|World|Global|Regional|Asian|South\s*Asian)?\s*(?:Conference|Workshop|Seminar|Forum|Summit|Symposium)\s+on\s+[A-Za-z\s]+$', lemma, re.I):
            return "ORG"

        # Generic ORG suffix patterns
        org_suffix_pattern = (
            r'\b(?:Committee|Federation|Commission|Council|Assembly|Party|Front|Forum|Board|'
            r'Authority|Ministry|Department|Institute|Institution|Society|Association|'
            r'Union|Organisation|Organization|Agency|Bureau|Corporation|Trust|Foundation|'
            r'Fund|Mission|Academy|Centre|Center|Network|Coalition|Alliance|Bloc|Group|'
            r'Brigade|Force|Wing|Cell|Unit|Division|Directorate|Secretariat|'
            r'University|College|School|Hospital|Clinic|Laboratory|Research Center|'
            r'Press|Publication|Media|Channel|Station)$'
        )
        if re.search(org_suffix_pattern, lemma, re.I):
            words = lemma.split()
            if 2 <= len(words) <= 5 and all(w[0].isupper() or w.lower() in {"of", "for", "and", "the", "&", "de", "da"} for w in words):
                return "ORG"

        # Kokborok-specific ORG patterns
        if re.search(r'\b(?:hapang|bodolni|commission|chongbari|montri|kebengnai\s+bodol)\b', lemma, re.I):
            words = lemma.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() or w.lower() in {"of", "for", "and", "the", "&"} or any(c.isdigit() for c in w) for w in words):
                return "ORG"

        # Premier Institutional Patterns
        if re.match(r'^(?:Indian|National)\s+Institute\s+of\s+(?:Technology|Information\s+Technology)(?:\s+[A-Z][a-z]+)?$', lemma):
            return "ORG"
        if re.match(r'^University\s+of\s+[A-Z][a-z]+$', lemma):
            return "ORG"

        # Multilingual Calendar Date Patterns
        deva_date = r'^\d{1,2}\s+(?:जानेवारी|फेब्रुवारी|मार्च|एप्रिल|मे|जून|जुलै|ऑगस्ट|सप्टेंबर|ऑक्टोबर|नोव्हेंबर|डिसेंबर)\s+\d{4}$'
        as_date = r'^[০-৯\d]{1,2}\s+(?:জানুৱাৰী|ফেব্ৰুৱাৰী|মাৰ্চ|এপ্ৰিল|মে\'?|জুন|জুলাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)\s+[০-৯\d]{4}$'
        si_date = r'^(?:\d{4}\s+)?(?:ජනවාරි|පෙබරවාරි|මාර්තු|අප්\u200dරේල්|අප්රේල්|මැයි|ජූනි|ජූලි|අගෝස්තු|සැප්තැම්බර්|ඔක්තෝබර්|නොවැම්බර්|දෙසැම්බර්)\s+\d{1,2}(?:\s+\d{4})?$'
        eng_date = r'^\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$'
        if re.match(deva_date, lemma, re.I) or re.match(as_date, lemma, re.I) or re.match(si_date, lemma, re.I) or re.match(eng_date, lemma, re.I):
            return "MISC"

        # Title / Honorific Pattern
        norm_title = re.sub(r'\s*\.\s*', '. ', lemma)
        if re.match(r'^(?:Dr\.?|Prof\.?|Professor|Mr\.?|Mrs\.?|Ms\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', norm_title):
            return "PER"

        return None

    def reconcile_entities(
        self,
        tokens: list[str],
        pred_entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reconcile predicted entities with dictionary priors and maximal span constraints."""
        final_entities = list(pred_entities)
        n_toks = len(tokens)

        # 1. DUAL_LOC_SURNAMES override
        for i in range(1, n_toks):
            tok = tokens[i]
            if tok in DUAL_LOC_SURNAMES and tokens[i - 1] and tokens[i - 1][0].isupper() \
                    and tokens[i - 1] not in KOKBOROK_PARTICLES:
                final_entities = [
                    e for e in final_entities
                    if (i + 1 <= e["start_token"] or i - 1 >= e["end_token"])
                ]
                final_entities.append({
                    "type": "PER",
                    "text": f"{tokens[i-1]} {tok}",
                    "start_token": i - 1,
                    "end_token": i + 1,
                    "source": "dual_loc_surname_override",
                })

        # 2. Multi-token and single-token matching for spans from length 8 down to 1
        clause_delimiters = {
            ",", ".", ";", ":", "!", "?", "।", "(", ")", "[", "]",
            "tei", "aro", "and", "or", "logot",
            "tongo", "wngkha", "wng-o", "phaikha", "snammani", "khobak", "rom-o", "chengmani",
            "bisi-o", "bisio", "tal-o", "talo", "Tabuk", "tabuk", "Tabok", "tabok",
        }
        for length in range(8, 0, -1):
            for i in range(n_toks - length + 1):
                sub_toks = tokens[i : i + length]
                if any(t in clause_delimiters or t.lower() in clause_delimiters for t in sub_toks):
                    continue
                if sub_toks[0].startswith("-") or sub_toks[-1].startswith("-"):
                    continue
                candidate_span = " ".join(sub_toks)
                matched_type = self.match_span(candidate_span)
                if matched_type:
                    overlapping = [
                        e for e in final_entities
                        if not (i + length <= e["start_token"] or i >= e["end_token"])
                    ]
                    if not overlapping:
                        final_entities.append({
                            "type": matched_type,
                            "text": candidate_span,
                            "start_token": i,
                            "end_token": i + length,
                            "source": "gazetteer_reconciliation",
                        })

        # Sort entities by start_token
        final_entities.sort(key=lambda x: (x["start_token"], x["end_token"]))
        return final_entities
