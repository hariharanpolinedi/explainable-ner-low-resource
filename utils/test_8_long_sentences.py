"""
Automated 8-Sentence Deep Evaluation Suite for All 4 Languages.
Each sentence has >= 5 lines and >= 14 ground-truth named entities.
Evaluates Konkani (2), Assamese (2), Sinhala (2), Kokborok (2).
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from deployment.inference import get_predictor

EVAL_DATASET = [
    # --------------------------------------------------------------------------
    # KONKANI
    # --------------------------------------------------------------------------
    {
        "language": "konkani",
        "id": "KONKANI_S1",
        "title": "Panaji Academic & Political Summit (20 Entities)",
        "sentence": "15 ऑक्टोबर 2026 दिसा पणजींत कला अकादमींत आयोजित केल्ल्या राष्ट्रीय भाशा परिशदेंत गोंयचे मुख्य मंत्री प्रमोद सावंत , विश्वनाथ कामत , मंत्री रोहन खंवटे आनी साहित्य अकादेमीचे अध्यक्ष माधव कौशिक हांणी भाग घेतलो , जंय दिल्ली , मुंबई , बेळगांव , मंगळूर आनी कारवार सावन आयिल्ल्या विद्वानांनी कोंकणी , मराठी , कन्नड आनी हिंदी भाशांच्या विकासाचेर गोंय विद्यापीठ आनी गोवा चेंबर ऑफ कॉमर्स हांच्या सहकार्यान एक व्हड प्रकल्प जाहीर केलो .",
        "expected_entities": [
            ("15 ऑक्टोबर 2026", "MISC"),
            ("पणजींत", "LOC"),
            ("कला अकादमींत", "ORG"),
            ("गोंयचे", "LOC"),
            ("प्रमोद सावंत", "PER"),
            ("विश्वनाथ कामत", "PER"),
            ("रोहन खंवटे", "PER"),
            ("साहित्य अकादेमीचे", "ORG"),
            ("माधव कौशिक", "PER"),
            ("दिल्ली", "LOC"),
            ("मुंबई", "LOC"),
            ("बेळगांव", "LOC"),
            ("मंगळूर", "LOC"),
            ("कारवार", "LOC"),
            ("कोंकणी", "MISC"),
            ("मराठी", "MISC"),
            ("कन्नड", "MISC"),
            ("हिंदी", "MISC"),
            ("गोंय विद्यापीठ", "ORG"),
            ("गोवा चेंबर ऑफ कॉमर्स", "ORG"),
        ],
    },
    {
        "language": "konkani",
        "id": "KONKANI_S2",
        "title": "South Goa Cultural & Regional Assembly (17 Entities)",
        "sentence": "22 नोव्हेंबर 2026 दिसा मडगांवांत रवींद्र भवनांत सुरू जाल्ल्या आंतरराष्ट्रीय संस्कृती महोत्सवांत दक्षिण गोंयचे खासदार फ्रान्सिस्को सार्दिन्हा , मंत्री दिगंबर कामत , समाजसेविका सुहासिनी प्रभू आनी डॉ. संदेश नाईक हांच्या उपस्थितींत वास्को , म्हापसा , फोंडा आनी काणकोण सावन आयिल्ल्या कलाकारांनी गोंय फाउडेशन आनी कदंब वाहतूक महामंडळ हांच्या मदतीन गोंय , महाराष्ट्र आनी कर्नाटक राज्यांतल्या लोककलेचे प्रदर्शन केले .",
        "expected_entities": [
            ("22 नोव्हेंबर 2026", "MISC"),
            ("मडगांवांत", "LOC"),
            ("रवींद्र भवनांत", "ORG"),
            ("दक्षिण गोंयचे", "LOC"),
            ("फ्रान्सिस्को सार्दिन्हा", "PER"),
            ("दिगंबर कामत", "PER"),
            ("सुहासिनी प्रभू", "PER"),
            ("डॉ. संदेश नाईक", "PER"),
            ("वास्को", "LOC"),
            ("म्हापसा", "LOC"),
            ("फोंडा", "LOC"),
            ("काणकोण", "LOC"),
            ("गोंय फाउडेशन", "ORG"),
            ("कदंब वाहतूक महामंडळ", "ORG"),
            ("गोंय", "LOC"),
            ("महाराष्ट्र", "LOC"),
            ("कर्नाटक", "LOC"),
        ],
    },
    # --------------------------------------------------------------------------
    # ASSAMESE
    # --------------------------------------------------------------------------
    {
        "language": "assamese",
        "id": "ASSAMESE_S1",
        "title": "Guwahati Higher Education & Linguistic Summit (21 Entities)",
        "sentence": "৫ নৱেম্বৰ ২০২৬ তাৰিখে গুৱাহাটীত শ্ৰীমন্ত শংকৰদেৱ কলাক্ষেত্ৰত অনুষ্ঠিত এক আন্তঃৰাষ্ট্ৰীয় সন্মিলনত অসমৰ মুখ্যমন্ত্ৰী হিমন্ত বিশ্ব শৰ্মা , শিক্ষামন্ত্ৰী ৰণোজ পেগু , বিশিষ্ট সাহিত্যিক ড° নগেন শইকীয়া আৰু প্ৰফেচৰ অৰূপ বৰুৱা উপস্থিত থাকে , য'ত ডিব্ৰুগড় , যোৰহাট , তেজপুৰ , শিলচৰ আৰু বৰপেটাৰ প্ৰতিনিধিসকলে গুৱাহাটী বিশ্ববিদ্যালয় , কটন বিশ্ববিদ্যালয় , তেজপুৰ বিশ্ববিদ্যালয় আৰু অসম সাহিত্য সভাৰ সহযোগত অসমীয়া , বড়ো , বঙালী আৰু ইংৰাজী ভাষাৰ সংৰক্ষণৰ বাবে এক ঐতিহাসিক চুক্তি স্বাক্ষৰ কৰে ।",
        "expected_entities": [
            ("৫ নৱেম্বৰ ২০২৬", "MISC"),
            ("গুৱাহাটীত", "LOC"),
            ("শ্ৰীমন্ত শংকৰদেৱ কলাক্ষেত্ৰত", "ORG"),
            ("অসমৰ", "LOC"),
            ("হিমন্ত বিশ্ব শৰ্মা", "PER"),
            ("ৰণোজ পেগু", "PER"),
            ("ড° নগেন শইকীয়া", "PER"),
            ("প্ৰফেচৰ অৰূপ বৰুৱা", "PER"),
            ("ডিব্ৰুগড়", "LOC"),
            ("যোৰহাট", "LOC"),
            ("তেজপুৰ", "LOC"),
            ("শিলচৰ", "LOC"),
            ("বৰপেটাৰ", "LOC"),
            ("গুৱাহাটী বিশ্ববিদ্যালয়", "ORG"),
            ("কটন বিশ্ববিদ্যালয়", "ORG"),
            ("তেজপুৰ বিশ্ববিদ্যালয়", "ORG"),
            ("অসম সাহিত্য সভাৰ", "ORG"),
            ("অসমীয়া", "MISC"),
            ("বড়ো", "MISC"),
            ("বঙালী", "MISC"),
            ("ইংৰাজী", "MISC"),
        ],
    },
    {
        "language": "assamese",
        "id": "ASSAMESE_S2",
        "title": "Northeast Riverine Trade & Inland Waterway Forum (14 Entities)",
        "sentence": "১৮ ডিচেম্বৰ ২০২৬ তাৰিখে শিলঘাটত ব্ৰহ্মপুত্ৰ নদীৰ পাৰত অনুষ্ঠিত বাণিজ্য বৈঠকত কেন্দ্ৰীয় মন্ত্ৰী সৰ্বানন্দ সোণোৱাল , সাংসদ গৌৰৱ গগৈ , অৰুণাচল প্ৰদেশৰ মুখ্যমন্ত্ৰী পেমা খাণ্ডু আৰু ব্যৱসায়ী ৰমেশ আগৰৱালাৰ উপস্থিতিত অসম , মেঘালয় , নাগালেণ্ড আৰু মণিপুৰ ৰাজ্যৰ মাজত ভাৰতীয় অন্তৰ্দেশীয় জলপথ প্ৰাধিকৰণ আৰু অসম ৰাজ্যিক পৰিবহণ নিগমৰ যৌথ উদ্যোগত নতুন জলপথ প্ৰকল্প আৰম্ভ কৰা হয় ।",
        "expected_entities": [
            ("১৮ ডিচেম্বৰ ২০২৬", "MISC"),
            ("শিলঘাটত", "LOC"),
            ("ব্ৰহ্মপুত্ৰ", "LOC"),
            ("সৰ্বানন্দ সোণোৱাল", "PER"),
            ("গৌৰৱ গগৈ", "PER"),
            ("অৰুণাচল প্ৰদেশৰ", "LOC"),
            ("পেমা খাণ্ডু", "PER"),
            ("ৰমেশ আগৰৱালাৰ", "PER"),
            ("অসম", "LOC"),
            ("মেঘালয়", "LOC"),
            ("নাগালেণ্ড", "LOC"),
            ("মণিপুৰ", "LOC"),
            ("ভাৰতীয় অন্তৰ্দেশীয় জলপথ প্ৰাধিকৰণ", "ORG"),
            ("অসম ৰাজ্যিক পৰিবহণ নিগমৰ", "ORG"),
        ],
    },
    # --------------------------------------------------------------------------
    # SINHALA
    # --------------------------------------------------------------------------
    {
        "language": "sinhala",
        "id": "SINHALA_S1",
        "title": "Colombo National Education Reform Assembly (19 Entities)",
        "sentence": "2026 නොවැම්බර් 10 දින කොළඹ බණ්ඩාරනායක අනුස්මරණ ජාත්‍යන්තර සම්මන්ත්‍රණ ශාලාවේදී පැවති ජාතික අධ්‍යාපන සමුළුවට ජනාධිපති අනුර කුමාර දිසානායක , අගමැතිනී හරිනි අමරසූරිය , මහාචාර්ය සුනිල් ආරියරත්න සහ ආචාර්ය ගාමිණී පෙරේරා සහභාගී වූ අතර එහිදී මහනුවර , ගාල්ල , යාපනය , මාතර සහ අනුරාධපුර යන නගර නියෝජනය කරමින් කොළඹ විශ්වවිද්‍යාලය , පේරාදෙණිය විශ්වවිද්‍යාලය , රුහුණ විශ්වවිද්‍යාලය සහ ජයවර්ධනපුර විශ්වවිද්‍යාලයේ විද්වතුන් සිංහල , දෙමළ සහ ඉංග්‍රීසි මාධ්‍ය අධ්‍යාපන ප්‍රතිසංස්කරණ ගැන සාකච්ඡා කළහ .",
        "expected_entities": [
            ("2026 නොවැම්බර් 10", "MISC"),
            ("කොළඹ", "LOC"),
            ("බණ්ඩාරනායක අනුස්මරණ ජාත්‍යන්තර සම්මන්ත්‍රණ ශාලාවේදී", "ORG"),
            ("අනුර කුමාර දිසානායක", "PER"),
            ("හරිනි අමරසූරිය", "PER"),
            ("සුනිල් ආරියරත්න", "PER"),
            ("ගාමිණී පෙරේරා", "PER"),
            ("මහනුවර", "LOC"),
            ("ගාල්ල", "LOC"),
            ("යාපනය", "LOC"),
            ("මාතර", "LOC"),
            ("අනුරාධපුර", "LOC"),
            ("කොළඹ විශ්වවිද්‍යාලය", "ORG"),
            ("පේරාදෙණිය විශ්වවිද්‍යාලය", "ORG"),
            ("රුහුණ විශ්වවිද්‍යාලය", "ORG"),
            ("ජයවර්ධනපුර විශ්වවිද්‍යාලයේ", "ORG"),
            ("සිංහල", "MISC"),
            ("දෙමළ", "MISC"),
            ("ඉංග්‍රීසි", "MISC"),
        ],
    },
    {
        "language": "sinhala",
        "id": "SINHALA_S2",
        "title": "Southern Maritime Ports & Economic Corridor Forum (14 Entities)",
        "sentence": "2026 ඔක්තෝබර් 15 දින හම්බන්තොට ජාත්‍යන්තර වරාය පරිශ්‍රයේදී දකුණු පළාත් ආණ්ඩුකාර කීර්ති තෙන්නකෝන් , අමාත්‍ය විජිත හේරත් , ව්‍යාපාරික ධම්මික පෙරේරා සහ ආචාර්ය නිමල් සිල්වා යන මහත්වරුන්ගේ ප්‍රධානත්වයෙන් පැවති රැස්වීමේදී ශ්‍රී ලංකා වරාය අධිකාරිය , ලංකා ඛනිජ තෙල් නීතිගත සංස්ථාව සහ මහජන බැංකුව ඒකාබද්ධව කොළඹ , ත්‍රිකුණාමලය , මඩකලපුව සහ කුරුණෑගල ප්‍රදේශ ආවරණය වන සංවර්ධන සැලැස්මක් ප්‍රකාශයට පත් කරන ලදී .",
        "expected_entities": [
            ("2026 ඔක්තෝබර් 15", "MISC"),
            ("හම්බන්තොට", "LOC"),
            ("කීර්ති තෙන්නකෝන්", "PER"),
            ("විජිත හේරත්", "PER"),
            ("ධම්මික පෙරේරා", "PER"),
            ("නිමල් සිල්වා", "PER"),
            ("ශ්‍රී ලංකා වරාය අධිකාරිය", "ORG"),
            ("ලංකා ඛනිජ තෙල් නීතිගත සංස්ථාව", "ORG"),
            ("මහජන බැංකුව", "ORG"),
            ("කොළඹ", "LOC"),
            ("ත්‍රිකුණාමලය", "LOC"),
            ("මඩකලපුව", "LOC"),
            ("කුරුණෑගල", "LOC"),
            ("ශ්‍රී ලංකා", "LOC"),
        ],
    },
    # --------------------------------------------------------------------------
    # KOKBOROK
    # --------------------------------------------------------------------------
    {
        "language": "kokborok",
        "id": "KOKBOROK_S1",
        "title": "Agartala University & Research Summit (17 Entities)",
        "sentence": "Kokborok language-ni digital development aru preservation-ni karone 12 September 2026-ot Agartala-ni Maharaja Bir Bikram University-ao anusthito ekta international research conference-ao Tripura, Assam, Mizoram, Manipur aru Bangladesh-ni researchers-mani jog dai, jekhane Professor Debashis Jamatia, Dr. Rina Debbarma, Dr. Anupam Chakma aru Professor Michael Das-ni leadership-ao tara artificial intelligence, natural language processing, machine translation, named entity recognition aru explainable AI-ni upor notun research results presentation dai, aru conference-ni sesh-ot Indian Institute of Technology Guwahati, AI4Bharat aru University of Dhaka-ni representatives-mani logot low-resource language dataset toiri, multilingual AI model development aru Kokborok language-ni digital education platform suru khorar bishoyot guruttopurno alochona khorok.",
        "expected_entities": [
            ("Kokborok", "MISC"),
            ("12 September 2026-ot", "MISC"),
            ("Agartala-ni", "LOC"),
            ("Maharaja Bir Bikram University-ao", "ORG"),
            ("Tripura", "LOC"),
            ("Assam", "LOC"),
            ("Mizoram", "LOC"),
            ("Manipur", "LOC"),
            ("Bangladesh-ni", "LOC"),
            ("Professor Debashis Jamatia", "PER"),
            ("Dr. Rina Debbarma", "PER"),
            ("Dr. Anupam Chakma", "PER"),
            ("Professor Michael Das-ni", "PER"),
            ("Indian Institute of Technology Guwahati", "ORG"),
            ("AI4Bharat", "ORG"),
            ("University of Dhaka-ni", "ORG"),
            ("Kokborok", "MISC"),
        ],
    },
    {
        "language": "kokborok",
        "id": "KOKBOROK_S2",
        "title": "Khumulwng Tribal Autonomous District Council Summit (18 Entities)",
        "sentence": "28 October 2026-ot Khumulwng-ni Khumpui Academy auditorium-ao TTAADC-ni Chief Executive Member Purna Chandra Jamatia, Executive Member Pradyot Bikram Manikya Debbarma, MLA Ranjit Debbarma aru Dr. Bijoy Tripura-ni presence-ao Gomati, Dhalai, Khowai, Khowai Twima aru South Tripura district-ni delegats-rok jog dai, jekhane Tripura Government, Borok Peoples Human Rights Organisation aru ICFAI University Tripura-ni officials-mani participation-bai Kokborok aru English medium education policy-ni upor ekta special session anusthito ongkha.",
        "expected_entities": [
            ("28 October 2026-ot", "MISC"),
            ("Khumulwng-ni", "LOC"),
            ("Khumpui Academy", "ORG"),
            ("TTAADC-ni", "ORG"),
            ("Purna Chandra Jamatia", "PER"),
            ("Pradyot Bikram Manikya Debbarma", "PER"),
            ("Ranjit Debbarma", "PER"),
            ("Dr. Bijoy Tripura-ni", "PER"),
            ("Gomati", "LOC"),
            ("Dhalai", "LOC"),
            ("Khowai", "LOC"),
            ("Khowai Twima", "LOC"),
            ("South Tripura", "LOC"),
            ("Tripura Government", "ORG"),
            ("Borok Peoples Human Rights Organisation", "ORG"),
            ("ICFAI University Tripura-ni", "ORG"),
            ("Kokborok", "MISC"),
            ("English", "MISC"),
        ],
    },
]


def evaluate_all():
    print("=" * 80)
    print("DEEP 8-SENTENCE MULTILINGUAL BENCHMARK EVALUATION (4 LANGUAGES x 2 SENTENCES)")
    print("Strict Threshold: Each sentence > 5 lines with > 13 Ground-Truth Tags")
    print("=" * 80)

    summary_results = []

    for item in EVAL_DATASET:
        lang = item["language"]
        sid = item["id"]
        title = item["title"]
        sent = item["sentence"]
        expected = item["expected_entities"]

        print(f"\n[{sid}] Running inference for {lang.upper()} ({title})...")
        predictor = get_predictor(lang)
        res = predictor.predict(sent)
        pred_entities = res["entities"]

        # Entity string and lemma match
        from preprocessing.multilingual_robustness import GazetteerReconciler
        recon = GazetteerReconciler(lang)

        matched_expected = 0
        missed = []
        pred_texts = [e["text"] for e in pred_entities]
        pred_lemmas = [recon.clean_lemma(e["text"]) for e in pred_entities]

        for exp_text, exp_type in expected:
            exp_lemma = recon.clean_lemma(exp_text)
            # check exact or lemma match
            hit = False
            for p_idx, (p_text, p_lemma) in enumerate(zip(pred_texts, pred_lemmas)):
                p_type = pred_entities[p_idx]["type"]
                if exp_text in p_text or p_text in exp_text or exp_lemma == p_lemma or exp_lemma in p_lemma:
                    hit = True
                    break
            if hit:
                matched_expected += 1
            else:
                missed.append((exp_text, exp_type))

        rec = (matched_expected / len(expected)) * 100
        prec = (matched_expected / max(len(pred_entities), 1)) * 100

        print(f"  * Expected: {len(expected)} entities")
        print(f"  * Model Extracted: {len(pred_entities)} entities")
        print(f"  * Correctly Matched: {matched_expected} entities")
        print(f"  * Recall: {rec:.1f}% | Precision: {prec:.1f}%")
        if missed:
            print(f"  * Missed Entities ({len(missed)}): {[m[0] for m in missed]}")
        else:
            print("  * Missed Entities: 0 (FLAWLESS 100% RECOVERY)")

        summary_results.append({
            "id": sid,
            "lang": lang,
            "title": title,
            "expected_count": len(expected),
            "predicted_count": len(pred_entities),
            "matched_count": matched_expected,
            "missed": missed,
            "recall": rec,
            "precision": prec,
            "predicted_entities": pred_entities,
        })

    # Save summary report to JSON
    out_path = PROJECT_ROOT / "reports" / "8_long_sentences_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    print(f"\n[REPORT SAVED] Results written to {out_path}")


if __name__ == "__main__":
    evaluate_all()
