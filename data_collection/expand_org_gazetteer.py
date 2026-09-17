"""
Phase 10/11 Patch: ORG Gazetteer Expansion Pipeline.

Collects and compiles authentic regional organization names for Konkani, Assamese,
and Sinhala across:
- Government ministries, secretariats, and state departments
- Universities, colleges, research institutes, and educational boards
- Municipal councils, panchayats, and civic corporations
- Public sector undertakings, corporations, and regulatory boards
- High courts, district courts, and tribunals
- Hospitals, medical councils, and public trusts
- Banks, financial institutions, and cooperatives
- Regional media houses, press clubs, and cultural academies
- Sports associations and non-governmental bodies

Brings ORG gazetteer volume from ~27 entries to 75+ entries per language,
eliminating the relative sparsity compared to PER and LOC gazetteers.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.normalize import normalize_text

ADDITIONAL_ORGS_KONKANI = [
    # Government Departments & Public Bodies
    "गोवा सचिवालय",
    "गोवा राज्य निवडणूक आयोग",
    "गोवा माहिती आयोग",
    "गोवा मानवाधिकार आयोग",
    "गोवा लोकसेवा आयोग",
    "गोवा राज्य महिला आयोग",
    "गोवा वन विकास महामंडळ",
    "गोवा खाण विकास महामंडळ",
    "गोवा औद्योगिक विकास महामंडळ",
    "गोवा समाजकल्याण मंडळ",
    "गोवा खादी आनी ग्रामोद्योग मंडळ",
    "गोवा राज्य सहकारी बँक",
    "गोवा राज्य सहकारी दूध उत्पादक संघ",
    "गोवा राज्य बाल हक्क संरक्षण आयोग",
    "गोवा राज्य जैवविविधता मंडळ",
    "गोवा नगर नियोजन खातें",
    "गोवा सार्वजनिक बांधकाम खातें",
    "गोवा जलस्रोत खातें",
    "गोवा वीज खातें",
    "गोवा वाहतूक खातें",
    # Courts & Legal Institutions
    "मुंबई उच्च न्यायालय खंडपीठ पणजी",
    "उत्तर गोवा जिल्हा न्यायालय",
    "दक्षिण गोवा जिल्हा न्यायालय",
    "गोवा राज्य ग्राहक तक्रार निवारण आयोग",
    "गोवा लोकायुक्त",
    # Educational & Academic Institutions
    "गोवा माध्यमिक आनी उच्च माध्यमिक शिक्षण मंडळ",
    "गोवा तंत्रशिक्षण मंडळ",
    "गोवा राज्य तंत्रज्ञान संस्था",
    "राष्ट्रीय तंत्रज्ञान संस्था गोवा",
    "गोवा व्यवस्थापन संस्था",
    "चौगुले महाविद्यालय मडगांव",
    "धेंपे कला आनी विज्ञान महाविद्यालय पणजी",
    "सेंट झेवियर्स महाविद्यालय म्हापश्यां",
    "कार्मेल महिला महाविद्यालय नुवें",
    "रोझरी कला आनी वाणिज्य महाविद्यालय नावेली",
    "गोवा औषधनिर्माणशास्त्र महाविद्यालय",
    "गोवा संगीत महाविद्यालय",
    "गोवा वास्तुशास्त्र महाविद्यालय",
    "गोवा होमियोपॅथिक वैद्यकीय महाविद्यालय",
    "गोवा दंत महाविद्यालय",
    # Cultural, Literary & Community Bodies
    "कोंकणी राज्यभाषा संयुक्त कृती समिती",
    "अखिल भारतीय सारस्वत परिशद",
    "गोवा कोंकणी लेखक संघ",
    "नाट्य अकादमी गोवा",
    "गोवा ललित कला अकादमी",
    "दस्तुकार मंडळ गोवा",
    "कोंकणी विश्वकोश केंद्र",
    "द गोवा हिंदू असोसिएशन",
    "इन्स्टिट्यूट मिनेझिस ब्रागांझा",
    "गोवा दैवज्ञ ब्राह्मण मंडळ",
    # Hospitals & Healthcare
    "गोवा वैद्यकीय महाविद्यालय आनी रुग्णालय बांबोळी",
    "उत्तर गोवा जिल्हा रुग्णालय म्हापश्यां",
    "दक्षिण गोवा जिल्हा रुग्णालय मडगांव",
    "आशियाई कर्करोग संस्था गोवा",
    # Media & Press Organizations
    "गोवा युनियन ऑफ जर्नलिस्ट्स",
    "गोवा प्रेस क्लब",
    "कोंकणी डिजिटल नेटवर्क",
    "प्रुदंत दूरदर्शन गोवा",
    "आकाशवाणी पणजी",
]

ADDITIONAL_ORGS_ASSAMESE = [
    # Government Departments & Public Bodies
    "অসম লোকসেৱা আয়োগ",
    "অসম তথ্য আয়োগ",
    "অসম ৰাজ্যিক নিৰ্বাচন আয়োগ",
    "অসম মানৱ অধিকাৰ আয়োগ",
    "অসম ৰাজ্যিক মহিলা আয়োগ",
    "অসম ৰাজ্যিক তথ্য প্ৰযুক্তি নিগম",
    "অসম বন উন্নয়ন নিগম",
    "অসম চাহ উন্নয়ন নিগম",
    "অসম ঔদ্যোগিক উন্নয়ন নিগম",
    "অসম গেছ কোম্পানী লিমিটেড",
    "অসম জলসিঞ্চন বিভাগ",
    "অসম লোকনিৰ্মাণ বিভাগ",
    "অসম শিক্ষা বিভাগ",
    "অসম স্বাস্থ্য সেৱা সঞ্চালকালয়",
    "অসম শক্তি বিতৰণ কোম্পানী",
    "অসম ৰাজ্যিক দুগ্ধ নিগম",
    "অসম সমবায় এপেক্স বেংক",
    "অসম গ্ৰামীণ বিকাশ বেংক",
    "গুৱাহাটী পৌৰ নিগম",
    "ডিব্ৰুগড় পৌৰসভা",
    # Courts & Legal Institutions
    "অসম ৰাজ্যিক আইন সেৱা প্ৰাধিকৰণ",
    "অসম প্ৰশাসনিক ন্যায়াধিকৰণ",
    "কামৰূপ মহানগৰ জিলা আদালত",
    "অসম উপভোক্তা ন্যায়ালয়",
    "অসম লোকায়ুক্ত",
    # Educational & Academic Institutions
    "অসম উচ্চতৰ মাধ্যমিক শিক্ষা সংসদ",
    "মাধ্যমিক শিক্ষা পৰিষদ অসম",
    "ভাৰতীয় প্ৰযুক্তিবিদ্যা প্ৰতিষ্ঠান গুৱাহাটী",
    "ৰাষ্ট্ৰীয় প্ৰযুক্তিবিদ্যা প্ৰতিষ্ঠান শিলচৰ",
    "অসম ৰাজ্যিক মুক্ত বিশ্ববিদ্যালয়",
    "বড়োলেণ্ড বিশ্ববিদ্যালয়",
    "মাধৱদেৱ বিশ্ববিদ্যালয়",
    "ভট্টদেৱ বিশ্ববিদ্যালয়",
    "ৰবীন্দ্ৰনাথ ঠাকুৰ বিশ্ববিদ্যালয়",
    "গুৱাহাটী চিকিৎসা মহাবিদ্যালয়",
    "অসম চিকিৎসা মহাবিদ্যালয় ডিব্ৰুগড়",
    "শিলচৰ চিকিৎসা মহাবিদ্যালয়",
    "যোৰহাট চিকিৎসা মহাবিদ্যালয়",
    "তেজপুৰ চিকিৎসা মহাবিদ্যালয়",
    "অসম অভিযান্ত্ৰিক মহাবিদ্যালয়",
    "যোৰহাট অভিযান্ত্ৰিক মহাবিদ্যালয়",
    "জগন্নাথ বৰুৱা মহাবিদ্যালয়",
    "বি বৰুৱা মহাবিদ্যালয়",
    "হণ্ডিক ছোৱালী মহাবিদ্যালয়",
    # Cultural, Literary & Community Bodies
    "অসম জাতীয়তাবাদী যুৱ ছাত্ৰ পৰিষদ",
    "বড়ো সাহিত্য সভা",
    "মিচিং দৃৰ্বি কৌবাং",
    "সদৌ অসম লেখিকা সমাৰোহ সমিতি",
    "শ্ৰীমন্ত শংকৰদেৱ কলাক্ষেত্ৰ সমাজ",
    "অসম প্ৰকাশন পৰিষদ",
    "গুৱাহাটী প্ৰেছ ক্লাব",
    "অসম ক্ৰিকেট সন্থা",
    "অসম ফুটবল সন্থা",
    # Media & Broadcasting
    "আকাশবাণী গুৱাহাটী",
    "দূৰদৰ্শন কেন্দ্ৰ গুৱাহাটী",
    "নিউজ১৮ অসম উত্তৰ-পূব",
    "ডিৱাই ৩৬৫",
    "এনডি২৪",
]

ADDITIONAL_ORGS_SINHALA = [
    # Ministries & Government Departments
    "ශ්‍රී ලංකා අධ්‍යාපන අමාත්‍යාංශය",
    "ශ්‍රී ලංකා සෞඛ්‍ය අමාත්‍යාංශය",
    "ශ්‍රී ලංකා ආරක්ෂක අමාත්‍යාංශය",
    "ශ්‍රී ලංකා මුදල් අමාත්‍යාංශය",
    "ශ්‍රී ලංකා කෘෂිකර්ම අමාත්‍යාංශය",
    "ශ්‍රී ලංකා ප්‍රවාහන අමාත්‍යාංශය",
    "ශ්‍රී ලංකා පරිසර අමාත්‍යාංශය",
    "ශ්‍රී ලංකා විදේශ කටයුතු අමාත්‍යාංශය",
    "ශ්‍රී ලංකා ආගමන විගමන දෙපාර්තමේන්තුව",
    "ශ්‍රී ලංකා රේගු දෙපාර්තමේන්තුව",
    "ශ්‍රී ලංකා දේශීය ආදායම් දෙපාර්තමේන්තුව",
    "ශ්‍රී ලංකා වන සංරක්ෂණ දෙපාර්තමේන්තුව",
    "ශ්‍රී ලංකා වනජීවී සංරක්ෂණ දෙපාර්තමේන්තුව",
    "ශ්‍රී ලංකා කාලගුණ විද්‍යා දෙපාර්තමේන්තුව",
    "ශ්‍රී ලංකා තැපැල් දෙපාර්තමේන්තුව",
    "කොළඹ මහා නගර සභාව",
    "මහනුවර මහා නගර සභාව",
    "ගාල්ල මහා නගර සභාව",
    # Commissions, Regulatory & Public Corporations
    "ශ්‍රී ලංකා මැතිවරණ කොමිසම",
    "ශ්‍රී ලංකා මානව හිමිකම් කොමිෂන් සභාව",
    "ශ්‍රී ලංකා අල්ලස් හෝ දූෂණ විමර්ශන කොමිසම",
    "ශ්‍රී ලංකා රාජ්‍ය සේවා කොමිෂන් සභාව",
    "ජාතික ජල සම්පාදන හා ජලාපවහන මණ්ඩලය",
    "ශ්‍රී ලංකා වරාය අධිකාරිය",
    "ශ්‍රී ලංකා ටෙලිකොම් ආයතනය",
    "ශ්‍රී ලංකා තෙල් සංස්ථාව",
    "ශ්‍රී ලංකා රක්ෂණ සංස්ථාව",
    "ජාතික ඉතිරිකිරීමේ බැංකුව",
    "හැටන් නැෂනල් බැංකුව",
    "කොමර්ෂල් බැංකුව",
    # Judicial & Legal Bodies
    "ශ්‍රී ලංකා ශ්‍රේෂ්ඨාධිකරණය",
    "ශ්‍රී ලංකා අභියාචනාධිකරණය",
    "කොළඹ මහාධිකරණය",
    "ශ්‍රී ලංකා නීතිඥ සංගමය",
    "නීතිපති දෙපාර්තමේන්තුව",
    # Universities & Higher Education Institutes
    "ශ්‍රී ලංකා විවෘත විශ්වවිද්‍යාලය",
    "වයඹ විශ්වවිද්‍යාලය",
    "සබරගමුව විශ්වවිද්‍යාලය",
    "රජරට විශ්වවිද්‍යාලය",
    "ඌව වෙල්ලස්ස විශ්වවිද්‍යාලය",
    "ශ්‍රී ලංකා සබරගමුව විශ්වවිද්‍යාලය",
    "ශ්‍රී ලංකා තොරතුරු තාක්ෂණ ආයතනය",
    "ශ්‍රීමත් ජෝන් කොතලාවල ආරක්ෂක විශ්වවිද්‍යාලය",
    "කොළඹ ජාතික රෝහල",
    "මහනුවර ශික්ෂණ රෝහල",
    "කරාපිටිය ශික්ෂණ රෝහල",
    "ශ්‍රී ජයවර්ධනපුර මහ රෝහල",
    # Cultural & Media
    "ශ්‍රී ලංකා කලා මණ්ඩලය",
    "ශ්‍රී ලංකා පුවත්පත් මණ්ඩලය",
    "ශ්‍රී ලංකා ගුවන් විදුලි සංස්ථාව",
    "ජාතික චිත්‍රපට සංස්ථාව",
    "ශ්‍රී ලංකා ක්‍රීඩා අමාත්‍යාංශය",
    "ශ්‍රී ලංකා ඔලිම්පික් කමිටුව",
]


def expand_gazetteers(data_dir: Path | str | None = None) -> dict[str, dict[str, int]]:
    """
    Expand data/gazetteers/{lang}/orgs.txt with verified institutional organizations.
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "gazetteers"
    else:
        data_dir = Path(data_dir)

    expansion_map = {
        "konkani": ADDITIONAL_ORGS_KONKANI,
        "assamese": ADDITIONAL_ORGS_ASSAMESE,
        "sinhala": ADDITIONAL_ORGS_SINHALA,
    }

    counts: dict[str, dict[str, int]] = {}

    for lang, add_list in expansion_map.items():
        org_file = data_dir / lang / "orgs.txt"
        if not org_file.is_file():
            raise FileNotFoundError(f"Missing orgs.txt for {lang} at {org_file}")

        with open(org_file, "r", encoding="utf-8") as f:
            existing = [line.strip() for line in f if line.strip()]

        before_cnt = len(existing)
        existing_set = set(normalize_text(e) for e in existing)

        added_cnt = 0
        for item in add_list:
            norm_item = normalize_text(item)
            if norm_item and norm_item not in existing_set:
                existing.append(norm_item)
                existing_set.add(norm_item)
                added_cnt += 1

        with open(org_file, "w", encoding="utf-8") as f:
            for item in existing:
                f.write(f"{item}\n")

        after_cnt = len(existing)
        counts[lang] = {
            "before": before_cnt,
            "added": added_cnt,
            "after": after_cnt,
        }
        print(f"[{lang.upper()}] Expanded orgs.txt: {before_cnt} -> {after_cnt} (+{added_cnt} entries)")

    return counts


if __name__ == "__main__":
    expand_gazetteers()
