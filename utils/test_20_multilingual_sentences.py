"""
Comprehensive 20-Sentence Multilingual NER Evaluation Benchmark.
Covers 4 languages x 5 long sentences = 20 sentences.
Each sentence contains 15+ ground truth entities.
Evaluates Sinhala, Konkani, Assamese, and Kokborok on GPU.
"""

import sys
import json
import re
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

DATASET_20 = [
    # =========================================================================
    # 1. SINHALA (5 Sentences)
    # =========================================================================
    {
        "language": "sinhala",
        "id": "SI_S1",
        "title": "Sinhala 1: Colombo University Summit",
        "sentence": "2026 සැප්තැම්බර් 18 වන දින කොළඹ විශ්වවිද්යාලයේ පැවති සමුළුවට මහාචාර්ය නදීකා පෙරේරා, ආචාර්ය කසුන් ජයවර්ධන, ආචාර්ය චතුරංග සිල්වා, සමන් කුමාර සහ රවිඳු සේනානායක සහභාගී වූ අතර කොළඹ, මහනුවර, ගාල්ල, යාපනය සහ අනුරාධපුරයෙන් පැමිණි පර්යේෂකයන් Google Research, Microsoft Research, AI4Bharat සහ මොරටුව විශ්වවිද්යාලය සමඟ කෘත්රිම බුද්ධිය සහ ස්වාභාවික භාෂා සැකසීම පිළිබඳ සාකච්ඡා කළහ.",
        "expected_entities": [
            ("2026 සැප්තැම්බර් 18", "MISC"),
            ("කොළඹ විශ්වවිද්යාලයේ", "ORG"),
            ("නදීකා පෙරේරා", "PER"),
            ("කසුන් ජයවර්ධන", "PER"),
            ("චතුරංග සිල්වා", "PER"),
            ("සමන් කුමාර", "PER"),
            ("රවිඳු සේනානායක", "PER"),
            ("කොළඹ", "LOC"),
            ("මහනුවර", "LOC"),
            ("ගාල්ල", "LOC"),
            ("යාපනය", "LOC"),
            ("අනුරාධපුරයෙන්", "LOC"),
            ("Google Research", "ORG"),
            ("Microsoft Research", "ORG"),
            ("AI4Bharat", "ORG"),
            ("මොරටුව විශ්වවිද්යාලය", "ORG"),
            ("කෘත්රිම බුද්ධිය", "MISC"),
            ("ස්වාභාවික භාෂා සැකසීම", "MISC"),
        ]
    },
    {
        "language": "sinhala",
        "id": "SI_S2",
        "title": "Sinhala 2: Moratuwa University Presentation",
        "sentence": "2027 ජනවාරි 12 වන දින මොරටුව විශ්වවිද්යාලයේදී ආචාර්ය අමන්දා ප්රනාන්දු, මහාචාර්ය සුනිල් පෙරේරා, නිමල් ජයසේකර සහ චතුරංග ද සිල්වා විසින් ශ්රී ලංකාව, ඉන්දියාව, බංග්ලාදේශය සහ නේපාලය නියෝජනය කළ පර්යේෂකයන් සමඟ සිංහල, දෙමළ, හින්දි සහ බෙංගාලි භාෂා සඳහා Named Entity Recognition, Machine Translation සහ Explainable AI පිළිබඳ පර්යේෂණ ඉදිරිපත් කරන ලදී.",
        "expected_entities": [
            ("2027 ජනවාරි 12", "MISC"),
            ("මොරටුව විශ්වවිද්යාලයේදී", "ORG"),
            ("අමන්දා ප්රනාන්දු", "PER"),
            ("සුනිල් පෙරේරා", "PER"),
            ("නිමල් ජයසේකර", "PER"),
            ("චතුරංග ද සිල්වා", "PER"),
            ("ශ්රී ලංකාව", "LOC"),
            ("ඉන්දියාව", "LOC"),
            ("බංග්ලාදේශය", "LOC"),
            ("නේපාලය", "LOC"),
            ("සිංහල", "MISC"),
            ("දෙමළ", "MISC"),
            ("හින්දි", "MISC"),
            ("බෙංගාලි", "MISC"),
            ("Named Entity Recognition", "MISC"),
            ("Machine Translation", "MISC"),
            ("Explainable AI", "MISC"),
        ]
    },
    {
        "language": "sinhala",
        "id": "SI_S3",
        "title": "Sinhala 3: BMICH Colombo Results",
        "sentence": "2027 මාර්තු 5 වන දින බණ්ඩාරනායක අනුස්මරණ ජාත්යන්තර සම්මන්ත්රණ ශාලාවේදී කොළඹ විශ්වවිද්යාලය, ජයවර්ධනපුර විශ්වවිද්යාලය, ශ්රී ලංකා තොරතුරු තාක්ෂණ ආයතනය, Google සහ NVIDIA හි පර්යේෂකයන් වන ආචාර්ය රිමාල් පෙරේරා, කවිඳු ජයවර්ධන, මාලිනී සිල්වා සහ සචින් ප්රනාන්දු විසින් සිංහල සහ දෙමළ Natural Language Processing ව්යාපෘති පිළිබඳ නව ප්රතිඵල ඉදිරිපත් කළහ.",
        "expected_entities": [
            ("2027 මාර්තු 5", "MISC"),
            ("බණ්ඩාරනායක අනුස්මරණ ජාත්යන්තර සම්මන්ත්රණ ශාලාවේදී", "ORG"),
            ("කොළඹ විශ්වවිද්යාලය", "ORG"),
            ("ජයවර්ධනපුර විශ්වවිද්යාලය", "ORG"),
            ("ශ්රී ලංකා තොරතුරු තාක්ෂණ ආයතනය", "ORG"),
            ("Google", "ORG"),
            ("NVIDIA", "ORG"),
            ("රිමාල් පෙරේරා", "PER"),
            ("කවිඳු ජයවර්ධන", "PER"),
            ("මාලිනී සිල්වා", "PER"),
            ("සචින් ප්රනාන්දු", "PER"),
            ("සිංහල", "MISC"),
            ("දෙමළ", "MISC"),
            ("Natural Language Processing", "MISC"),
        ]
    },
    {
        "language": "sinhala",
        "id": "SI_S4",
        "title": "Sinhala 4: Jaffna University Workshop",
        "sentence": "2027 අප්රේල් 22 වන දින යාපනය විශ්වවිද්යාලයේ පැවති තාක්ෂණික වැඩමුළුවකට මහාචාර්ය අරුණ කුමාර, ආචාර්ය දිනේෂ් රාජපක්ෂ, රුවින් ජයවර්ධන සහ නදීෂා පෙරේරා සමඟ ඉන්දියාවේ චෙන්නායි, බැංගලෝර් සහ නවදිල්ලියෙන් පැමිණි පර්යේෂකයන් සහභාගී වී Artificial Intelligence, Machine Learning, Computer Vision සහ Multilingual NLP පිළිබඳ පුහුණුවක් ලබා දුන්හ.",
        "expected_entities": [
            ("2027 අප්රේල් 22", "MISC"),
            ("යාපනය විශ්වවිද්යාලයේ", "ORG"),
            ("අරුණ කුමාර", "PER"),
            ("දිනේෂ් රාජපක්ෂ", "PER"),
            ("රුවින් ජයවර්ධන", "PER"),
            ("නදීෂා පෙරේරා", "PER"),
            ("ඉන්දියාවේ", "LOC"),
            ("චෙන්නායි", "LOC"),
            ("බැංගලෝර්", "LOC"),
            ("නවදිල්ලියෙන්", "LOC"),
            ("Artificial Intelligence", "MISC"),
            ("Machine Learning", "MISC"),
            ("Computer Vision", "MISC"),
            ("Multilingual NLP", "MISC"),
        ]
    },
    {
        "language": "sinhala",
        "id": "SI_S5",
        "title": "Sinhala 5: Galle Joint Assembly",
        "sentence": "2027 ජූනි 10 වන දින ගාල්ලේදී ශ්රී ලංකා රජය, කොළඹ විශ්වවිද්යාලය, මොරටුව විශ්වවිද්යාලය, AI4Bharat, University of Dhaka සහ Indian Institute of Technology Madras එක්ව සංවිධානය කළ සමුළුවේදී ආචාර්ය රිෂාන් ප්රනාන්දු, මහාචාර්ය කසුන් සිල්වා, අනුරාධා පෙරේරා සහ සුරංග ජයසේකර විසින් සිංහල, බෙංගාලි, හින්දි සහ දෙමළ භාෂාවල අඩු සම්පත් දත්ත කට්ටල පිළිබඳ පර්යේෂණ ඉදිරිපත් කළහ.",
        "expected_entities": [
            ("2027 ජූනි 10", "MISC"),
            ("ගාල්ලේදී", "LOC"),
            ("ශ්රී ලංකා රජය", "ORG"),
            ("කොළඹ විශ්වවිද්යාලය", "ORG"),
            ("මොරටුව විශ්වවිද්යාලය", "ORG"),
            ("AI4Bharat", "ORG"),
            ("University of Dhaka", "ORG"),
            ("Indian Institute of Technology Madras", "ORG"),
            ("රිෂාන් ප්රනාන්දු", "PER"),
            ("කසුන් සිල්වා", "PER"),
            ("අනුරාධා පෙරේරා", "PER"),
            ("සුරංග ජයසේකර", "PER"),
            ("සිංහල", "MISC"),
            ("බෙංගාලි", "MISC"),
            ("හින්දි", "MISC"),
            ("දෙමළ", "MISC"),
        ]
    },

    # =========================================================================
    # 2. KONKANI (5 Sentences)
    # =========================================================================
    {
        "language": "konkani",
        "id": "KO_S1",
        "title": "Konkani 1: Panaji International Conference",
        "sentence": "18 सप्टेंबर 2026 दिसा पणजींत गोंय विद्यापीठांत जाल्ल्या आंतरराष्ट्रीय परिषदेंत प्रा. अनिल नायक, डॉ. कविता काकोडकर, रोहन शेट, माया नाईक आनी सुरज गांवकर हांणी भारत, श्रीलंका, बांगलादेश आनी नेपाळांतल्यान आयिल्ल्या संशोधकां कडेन कोंकणी, मराठी, हिंदी आनी इंग्लीश भाशे खातीर Natural Language Processing, Named Entity Recognition आनी Artificial Intelligence विशीं चर्चा केली.",
        "expected_entities": [
            ("18 सप्टेंबर 2026", "MISC"),
            ("पणजींत", "LOC"),
            ("गोंय विद्यापीठांत", "ORG"),
            ("अनिल नायक", "PER"),
            ("कविता काकोडकर", "PER"),
            ("रोहन शेट", "PER"),
            ("माया नाईक", "PER"),
            ("सुरज गांवकर", "PER"),
            ("भारत", "LOC"),
            ("श्रीलंका", "LOC"),
            ("बांगलादेश", "LOC"),
            ("नेपाळांतल्यान", "LOC"),
            ("कोंकणी", "MISC"),
            ("मराठी", "MISC"),
            ("हिंदी", "MISC"),
            ("इंग्लीश", "MISC"),
            ("Natural Language Processing", "MISC"),
            ("Named Entity Recognition", "MISC"),
            ("Artificial Intelligence", "MISC"),
        ]
    },
    {
        "language": "konkani",
        "id": "KO_S2",
        "title": "Konkani 2: IIT Bombay & Goa University Technology Summit",
        "sentence": "12 जानेवारी 2027 दिसा गोंय विद्यापीठ, IIT Bombay, Google Research, Microsoft Research आनी AI4Bharat हांणी पणजींत आयोजित केल्ल्या तंत्रज्ञान परिषदेंत डॉ. रोहित कामत, प्रिया नाईक, अमित शेट, कविता फर्नांडिस आनी राहुल गांवकर हांणी गोंय, महाराष्ट्र, कर्नाटक, केरळ आनी नवी दिल्लींतल्यान आयिल्ल्या विद्यार्थ्यांक Machine Learning, Computer Vision, Machine Translation आनी Explainable AI विशीं प्रशिक्षण दिलें.",
        "expected_entities": [
            ("12 जानेवारी 2027", "MISC"),
            ("गोंय विद्यापीठ", "ORG"),
            ("IIT Bombay", "ORG"),
            ("Google Research", "ORG"),
            ("Microsoft Research", "ORG"),
            ("AI4Bharat", "ORG"),
            ("पणजींत", "LOC"),
            ("रोहित कामत", "PER"),
            ("प्रिया नाईक", "PER"),
            ("अमित शेट", "PER"),
            ("कविता फर्नांडिस", "PER"),
            ("राहुल गांवकर", "PER"),
            ("गोंय", "LOC"),
            ("महाराष्ट्र", "LOC"),
            ("कर्नाटक", "LOC"),
            ("केरळ", "LOC"),
            ("नवी दिल्लींतल्यान", "LOC"),
            ("Machine Learning", "MISC"),
            ("Computer Vision", "MISC"),
            ("Machine Translation", "MISC"),
            ("Explainable AI", "MISC"),
        ]
    },
    {
        "language": "konkani",
        "id": "KO_S3",
        "title": "Konkani 3: Margao Konkani Bhasha Mandal Program",
        "sentence": "5 मार्च 2027 दिसा मडगांवांत कोंकणी भाशा मंडळान आयोजीत केल्ल्या संशोधन कार्यक्रमांत प्राध्यापक संजय नायक, डॉ. रिमा शिरोडकर, अमोल काकोडकर, निशा राणे आनी सुमित फर्नांडिस हांणी कोंकणी, मराठी, हिंदी, कन्नड आनी मल्याळम भाशांचें multilingual dataset तयार करप, annotation करप आनी Named Entity Recognition मॉडेल तपासप ह्या विशयांचेर सादरीकरण केलें.",
        "expected_entities": [
            ("5 मार्च 2027", "MISC"),
            ("मडगांवांत", "LOC"),
            ("कोंकणी भाशा मंडळान", "ORG"),
            ("संजय नायक", "PER"),
            ("रिमा शिरोडकर", "PER"),
            ("अमोल काकोडकर", "PER"),
            ("निशा राणे", "PER"),
            ("सुमित फर्नांडिस", "PER"),
            ("कोंकणी", "MISC"),
            ("मराठी", "MISC"),
            ("हिंदी", "MISC"),
            ("कन्नड", "MISC"),
            ("मल्याळम", "MISC"),
            ("Named Entity Recognition", "MISC"),
        ]
    },
    {
        "language": "konkani",
        "id": "KO_S4",
        "title": "Konkani 4: IIT Delhi New Delhi Colloquium",
        "sentence": "22 एप्रिल 2027 दिसा नवी दिल्लींत Indian Institute of Technology Delhi, Ministry of Electronics and Information Technology, Google, NVIDIA आनी AI4Bharat हांच्या प्रतिनिधींनी डॉ. राजेश नाईक, माया शेट, रोहन कामत आनी स्नेहा गांवकर हांका भारत, गोंय, महाराष्ट्र, कर्नाटक आनी आसामांतल्या low-resource language research खातीर नव्या Artificial Intelligence प्रकल्पांचेर मार्गदर्शन दिलें.",
        "expected_entities": [
            ("22 एप्रिल 2027", "MISC"),
            ("नवी दिल्लींत", "LOC"),
            ("Indian Institute of Technology Delhi", "ORG"),
            ("Ministry of Electronics and Information Technology", "ORG"),
            ("Google", "ORG"),
            ("NVIDIA", "ORG"),
            ("AI4Bharat", "ORG"),
            ("राजेश नाईक", "PER"),
            ("माया शेट", "PER"),
            ("रोहन कामत", "PER"),
            ("स्नेहा गांवकर", "PER"),
            ("भारत", "LOC"),
            ("गोंय", "LOC"),
            ("महाराष्ट्र", "LOC"),
            ("कर्नाटक", "LOC"),
            ("आसामांतल्या", "LOC"),
            ("Artificial Intelligence", "MISC"),
        ]
    },
    {
        "language": "konkani",
        "id": "KO_S5",
        "title": "Konkani 5: Panaji Multilingual NLP Assembly",
        "sentence": "10 जून 2027 दिसा पणजींत गोंय विद्यापीठ, अखिल भारतीय कोंकणी परिषद, IIT Bombay, University of Colombo आनी University of Dhaka हांणी एकठांय घडयल्ल्या आंतरराष्ट्रीय परिषदेंत प्रा. सुनील कामत, डॉ. अंजली नाईक, विकास शेट, रवींद्र काकोडकर आनी नेहा फर्नांडिस हांणी कोंकणी, सिंहला, बंगाली, हिंदी आनी इंग्लीश भाशांत machine translation, sentiment analysis, information extraction आनी explainable artificial intelligence विशीं संशोधन सादर केलें.",
        "expected_entities": [
            ("10 जून 2027", "MISC"),
            ("पणजींत", "LOC"),
            ("गोंय विद्यापीठ", "ORG"),
            ("अखिल भारतीय कोंकणी परिषद", "ORG"),
            ("IIT Bombay", "ORG"),
            ("University of Colombo", "ORG"),
            ("University of Dhaka", "ORG"),
            ("सुनील कामत", "PER"),
            ("अंजली नाईक", "PER"),
            ("विकास शेट", "PER"),
            ("रवींद्र काकोडकर", "PER"),
            ("नेहा फर्नांडिस", "PER"),
            ("कोंकणी", "MISC"),
            ("सिंहला", "MISC"),
            ("बंगाली", "MISC"),
            ("हिंदी", "MISC"),
            ("इंग्लीश", "MISC"),
            ("machine translation", "MISC"),
            ("sentiment analysis", "MISC"),
            ("information extraction", "MISC"),
            ("explainable artificial intelligence", "MISC"),
        ]
    },

    # =========================================================================
    # 3. ASSAMESE (5 Sentences)
    # =========================================================================
    {
        "language": "assamese",
        "id": "AS_S1",
        "title": "Assamese 1: Gauhati University Summit",
        "sentence": "১৮ ছেপ্টেম্বৰ ২০২৬ তাৰিখে গুৱাহাটী বিশ্ববিদ্যালয়ত অনুষ্ঠিত আন্তঃৰাষ্ট্ৰীয় সন্মিলনত অধ্যাপক অনুপম শইকীয়া, ড° ৰীণা দত্ত, ৰাহুল বৰা, মায়া গগৈ আৰু অমিত কলিতাই ভাৰত, বাংলাদেশ, নেপাল আৰু শ্ৰীলংকাৰ পৰা অহা গৱেষকসকলৰ সৈতে অসমীয়া, বাংলা, হিন্দী আৰু ইংৰাজী ভাষাৰ Natural Language Processing, Named Entity Recognition আৰু Explainable AI সম্পৰ্কে আলোচনা কৰে।",
        "expected_entities": [
            ("১৮ ছেপ্টেম্বৰ ২০২৬", "MISC"),
            ("গুৱাহাটী বিশ্ববিদ্যালয়ত", "ORG"),
            ("অনুপম শইকীয়া", "PER"),
            ("ৰীণা দত্ত", "PER"),
            ("ৰাহুল বৰা", "PER"),
            ("মায়া গগৈ", "PER"),
            ("অমিত কলিতাই", "PER"),
            ("ভাৰত", "LOC"),
            ("বাংলাদেশ", "LOC"),
            ("নেপাল", "LOC"),
            ("শ্ৰীলংকাৰ", "LOC"),
            ("অসমীয়া", "MISC"),
            ("বাংলা", "MISC"),
            ("হিন্দী", "MISC"),
            ("ইংৰাজী", "MISC"),
            ("Natural Language Processing", "MISC"),
            ("Named Entity Recognition", "MISC"),
            ("Explainable AI", "MISC"),
        ]
    },
    {
        "language": "assamese",
        "id": "AS_S2",
        "title": "Assamese 2: IIT Guwahati & Tezpur Workshop",
        "sentence": "১২ জানুৱাৰী ২০২৭ তাৰিখে Indian Institute of Technology Guwahati, Tezpur University, AI4Bharat, Google Research আৰু Microsoft Research-এ গুৱাহাটীত আয়োজন কৰা কৰ্মশালাত ড° অনিল দাস, প্ৰিয়া শইকীয়া, সুমন বৰ্মন, ৰীতা গগৈ আৰু বিকাশ কলিতাই অসম, ত্ৰিপুৰা, মণিপুৰ, মিজোৰাম আৰু মেঘালয়ৰ শিক্ষাৰ্থীসকলক Artificial Intelligence, Machine Learning আৰু Computer Vision সম্পৰ্কে প্ৰশিক্ষণ দিয়ে।",
        "expected_entities": [
            ("১২ জানুৱাৰী ২০২৭", "MISC"),
            ("Indian Institute of Technology Guwahati", "ORG"),
            ("Tezpur University", "ORG"),
            ("AI4Bharat", "ORG"),
            ("Google Research", "ORG"),
            ("Microsoft Research", "ORG"),
            ("গুৱাহাটীত", "LOC"),
            ("অনিল দাস", "PER"),
            ("প্ৰিয়া শইকীয়া", "PER"),
            ("সুমন বৰ্মন", "PER"),
            ("ৰীতা গগৈ", "PER"),
            ("বিকাশ কলিতাই", "PER"),
            ("অসম", "LOC"),
            ("ত্ৰিপুৰা", "LOC"),
            ("মণিপুৰ", "LOC"),
            ("মিজোৰাম", "LOC"),
            ("মেঘালয়ৰ", "LOC"),
            ("Artificial Intelligence", "MISC"),
            ("Machine Learning", "MISC"),
            ("Computer Vision", "MISC"),
        ]
    },
    {
        "language": "assamese",
        "id": "AS_S3",
        "title": "Assamese 3: Tezpur University Multilingual Research",
        "sentence": "৫ মাৰ্চ ২০২৭ তাৰিখে তেজপুৰ বিশ্ববিদ্যালয়ত অধ্যাপক দেৱজিত শৰ্মা, ড° নন্দিতা বৰা, অমিত গগৈ, ৰোহিত দাস আৰু কাব্য শইকীয়াই Assam Literary Society, AI4Bharat, Indian Institute of Technology Guwahati আৰু Google Research-ৰ গৱেষকসকলৰ সৈতে অসমীয়া, বড়ো, বাংলা আৰু হিন্দী ভাষাৰ multilingual dataset, machine translation আৰু information extraction সম্পৰ্কে নতুন গৱেষণা উপস্থাপন কৰে।",
        "expected_entities": [
            ("৫ মাৰ্চ ২০২৭", "MISC"),
            ("তেজপুৰ বিশ্ববিদ্যালয়ত", "ORG"),
            ("দেৱজিত শৰ্মা", "PER"),
            ("নন্দিতা বৰা", "PER"),
            ("অমিত গগৈ", "PER"),
            ("ৰোহিত দাস", "PER"),
            ("কাব্য শইকীয়াই", "PER"),
            ("Assam Literary Society", "ORG"),
            ("AI4Bharat", "ORG"),
            ("Indian Institute of Technology Guwahati", "ORG"),
            ("Google Research", "ORG"),
            ("অসমীয়া", "MISC"),
            ("বড়ো", "MISC"),
            ("বাংলা", "MISC"),
            ("হিন্দী", "MISC"),
            ("machine translation", "MISC"),
            ("information extraction", "MISC"),
        ]
    },
    {
        "language": "assamese",
        "id": "AS_S4",
        "title": "Assamese 4: Rabindra Bhavan Technology Forum",
        "sentence": "২২ এপ্ৰিল ২০২৭ তাৰিখে ৰবীন্দ্ৰ ভৱনত অনুষ্ঠিত ভাষা প্ৰযুক্তি সন্মিলনত ড° মৃণাল দত্ত, অধ্যাপক সীমা কলিতা, ৰঞ্জিত বৰা, নেহা গগৈ আৰু প্ৰণৱ শইকীয়াই গুৱাহাটী, যোৰহাট, ডিব্ৰুগড়, মাজুলী আৰু শিলঘাটৰ পৰা অহা গৱেষকসকলৰ সৈতে অসমীয়া ভাষাৰ Named Entity Recognition, Sentiment Analysis, Machine Learning আৰু Explainable Artificial Intelligence সম্পৰ্কে আলোচনা কৰে।",
        "expected_entities": [
            ("২২ এপ্ৰিল ২০২৭", "MISC"),
            ("ৰবীন্দ্ৰ ভৱনত", "ORG"),
            ("মৃণাল দত্ত", "PER"),
            ("সীমা কলিতা", "PER"),
            ("ৰঞ্জিত বৰা", "PER"),
            ("নেহা গগৈ", "PER"),
            ("প্ৰণৱ শইকীয়াই", "PER"),
            ("গুৱাহাটী", "LOC"),
            ("যোৰহাট", "LOC"),
            ("ডিব্ৰুগড়", "LOC"),
            ("মাজুলী", "LOC"),
            ("শিলঘাটৰ", "LOC"),
            ("অসমীয়া", "MISC"),
            ("Named Entity Recognition", "MISC"),
            ("Sentiment Analysis", "MISC"),
            ("Machine Learning", "MISC"),
            ("Explainable Artificial Intelligence", "MISC"),
        ]
    },
    {
        "language": "assamese",
        "id": "AS_S5",
        "title": "Assamese 5: Guwahati Mega Research Conference",
        "sentence": "১০ জুন ২০২৭ তাৰিখে ভাৰত চৰকাৰ, গৌহাটী বিশ্ববিদ্যালয়, তেজপুৰ বিশ্ববিদ্যালয়, AI4Bharat, University of Dhaka আৰু Indian Institute of Technology Madras-ৰ সহযোগিতাত গুৱাহাটীত অনুষ্ঠিত বৃহৎ গৱেষণা সন্মিলনত অধ্যাপক সুশান্ত বৰ্মন, ড° ৰিমা দাস, অনুপম কলিতা, প্ৰিয়া গগৈ আৰু ৰাহুল শইকীয়াই অসমীয়া, বাংলা, হিন্দী, তামিল আৰু তেলুগু ভাষাৰ low-resource NLP, multilingual NER আৰু attention-based explainability সম্পৰ্কে গৱেষণা ফলাফল উপস্থাপন কৰে।",
        "expected_entities": [
            ("১০ জুন ২০২৭", "MISC"),
            ("ভাৰত চৰকাৰ", "ORG"),
            ("গৌহাটী বিশ্ববিদ্যালয়", "ORG"),
            ("তেজপুৰ বিশ্ববিদ্যালয়", "ORG"),
            ("AI4Bharat", "ORG"),
            ("University of Dhaka", "ORG"),
            ("Indian Institute of Technology Madras", "ORG"),
            ("গুৱাহাটীত", "LOC"),
            ("সুশান্ত বৰ্মন", "PER"),
            ("ৰিমা দাস", "PER"),
            ("অনুপম কলিতা", "PER"),
            ("প্ৰিয়া গগৈ", "PER"),
            ("ৰাহুল শইকীয়াই", "PER"),
            ("অসমীয়া", "MISC"),
            ("বাংলা", "MISC"),
            ("হিন্দী", "MISC"),
            ("তামিল", "MISC"),
            ("তেলুগু", "MISC"),
            ("multilingual NER", "MISC"),
        ]
    },

    # =========================================================================
    # 4. KOKBOROK (5 Sentences)
    # =========================================================================
    {
        "language": "kokborok",
        "id": "KB_S1",
        "title": "Kokborok 1: Tripura University International Conference",
        "sentence": "18 September 2026-ot Agartala-ao Tripura University-ao anusthito international conference-ao Professor Debashis Jamatia, Dr. Rina Debbarma, Dr. Anupam Chakma, Suman Reang aru Priya Jamatia-ni Tripura, Assam, Mizoram, Manipur aru Bangladesh-ni researchers-mani logot Kokborok, Assamese, Bengali aru Hindi language-ni Natural Language Processing, Named Entity Recognition aru Explainable AI-ni upor research discussion khorok.",
        "expected_entities": [
            ("18 September 2026", "MISC"),
            ("Agartala", "LOC"),
            ("Tripura University", "ORG"),
            ("Professor Debashis Jamatia", "PER"),
            ("Dr. Rina Debbarma", "PER"),
            ("Dr. Anupam Chakma", "PER"),
            ("Suman Reang", "PER"),
            ("Priya Jamatia", "PER"),
            ("Tripura", "LOC"),
            ("Assam", "LOC"),
            ("Mizoram", "LOC"),
            ("Manipur", "LOC"),
            ("Bangladesh", "LOC"),
            ("Kokborok", "MISC"),
            ("Assamese", "MISC"),
            ("Bengali", "MISC"),
            ("Hindi", "MISC"),
            ("Natural Language Processing", "MISC"),
            ("Named Entity Recognition", "MISC"),
            ("Explainable AI", "MISC"),
        ]
    },
    {
        "language": "kokborok",
        "id": "KB_S2",
        "title": "Kokborok 2: MBB University Training Program",
        "sentence": "12 January 2027-ot Agartala-ao Maharaja Bir Bikram University-ao AI4Bharat, Indian Institute of Technology Guwahati, Google Research aru Microsoft Research-ni collaboration-ao Dr. Rakesh Debbarma, Professor Maya Jamatia, Rahul Reang, Ananya Chakma aru Suman Debbarma-ni Tripura, Assam, Meghalaya, Manipur aru Mizoram-ni students-mani karone Artificial Intelligence, Machine Learning, Computer Vision aru Machine Translation-ni upor training program organize khorok.",
        "expected_entities": [
            ("12 January 2027", "MISC"),
            ("Agartala", "LOC"),
            ("Maharaja Bir Bikram University", "ORG"),
            ("AI4Bharat", "ORG"),
            ("Indian Institute of Technology Guwahati", "ORG"),
            ("Google Research", "ORG"),
            ("Microsoft Research", "ORG"),
            ("Dr. Rakesh Debbarma", "PER"),
            ("Professor Maya Jamatia", "PER"),
            ("Rahul Reang", "PER"),
            ("Ananya Chakma", "PER"),
            ("Suman Debbarma", "PER"),
            ("Tripura", "LOC"),
            ("Assam", "LOC"),
            ("Meghalaya", "LOC"),
            ("Manipur", "LOC"),
            ("Mizoram", "LOC"),
            ("Artificial Intelligence", "MISC"),
            ("Machine Learning", "MISC"),
            ("Computer Vision", "MISC"),
            ("Machine Translation", "MISC"),
        ]
    },
    {
        "language": "kokborok",
        "id": "KB_S3",
        "title": "Kokborok 3: Dharmanagar Language Workshop",
        "sentence": "5 March 2027-ot Dharmanagar-ao anusthito language technology workshop-ao Professor Sushanta Debbarma, Dr. Rina Chakma, Amit Jamatia, Priya Reang aru David Debbarma-ni Tripura University, AI4Bharat, Google Research aru Indian Institute of Technology Guwahati-ni researchers-mani logot Kokborok, Assamese, Bengali, Hindi aru English language-ni multilingual dataset, sentiment analysis, information extraction aru Named Entity Recognition-ni bishoyot notun research presentation dai.",
        "expected_entities": [
            ("5 March 2027", "MISC"),
            ("Dharmanagar", "LOC"),
            ("Professor Sushanta Debbarma", "PER"),
            ("Dr. Rina Chakma", "PER"),
            ("Amit Jamatia", "PER"),
            ("Priya Reang", "PER"),
            ("David Debbarma", "PER"),
            ("Tripura University", "ORG"),
            ("AI4Bharat", "ORG"),
            ("Google Research", "ORG"),
            ("Indian Institute of Technology Guwahati", "ORG"),
            ("Kokborok", "MISC"),
            ("Assamese", "MISC"),
            ("Bengali", "MISC"),
            ("Hindi", "MISC"),
            ("English", "MISC"),
            ("sentiment analysis", "MISC"),
            ("information extraction", "MISC"),
            ("Named Entity Recognition", "MISC"),
        ]
    },
    {
        "language": "kokborok",
        "id": "KB_S4",
        "title": "Kokborok 4: IIT Delhi Joint Research Meeting",
        "sentence": "22 April 2027-ot New Delhi-ao Indian Institute of Technology Delhi-ao Ministry of Electronics and Information Technology, Google Research, NVIDIA aru AI4Bharat-ni representatives-mani Professor Michael Jamatia, Dr. Anupam Debbarma, Riya Chakma, Rahul Reang aru Nandita Jamatia-ni logot Tripura, Assam, Meghalaya, West Bengal aru Bangladesh-ni low-resource language research-ni karone artificial intelligence aru multilingual NLP project-ni upor alochona khorok.",
        "expected_entities": [
            ("22 April 2027", "MISC"),
            ("New Delhi", "LOC"),
            ("Indian Institute of Technology Delhi", "ORG"),
            ("Ministry of Electronics and Information Technology", "ORG"),
            ("Google Research", "ORG"),
            ("NVIDIA", "ORG"),
            ("AI4Bharat", "ORG"),
            ("Professor Michael Jamatia", "PER"),
            ("Dr. Anupam Debbarma", "PER"),
            ("Riya Chakma", "PER"),
            ("Rahul Reang", "PER"),
            ("Nandita Jamatia", "PER"),
            ("Tripura", "LOC"),
            ("Assam", "LOC"),
            ("Meghalaya", "LOC"),
            ("West Bengal", "LOC"),
            ("Bangladesh", "LOC"),
            ("artificial intelligence", "MISC"),
            ("multilingual NLP", "MISC"),
        ]
    },
    {
        "language": "kokborok",
        "id": "KB_S5",
        "title": "Kokborok 5: Agartala Large-Scale Collaborative Project",
        "sentence": "10 June 2027-ot Agartala-ao Tripura University-ao Professor Debashis Jamatia, Dr. Rina Debbarma, Dr. Anupam Chakma, Professor Michael Das, Suman Reang aru Priya Jamatia-ni Tripura, Assam, Mizoram, Manipur, Meghalaya, Bangladesh aru New Delhi-ni researchers-mani logot AI4Bharat, Google Research, Microsoft Research, University of Dhaka aru Indian Institute of Technology Madras-ni collaboration-ao Kokborok, Assamese, Bengali, Hindi, Tamil aru Telugu language-ni multilingual Named Entity Recognition, Machine Translation, Attention Attribution aru Explainable Artificial Intelligence-ni upor large-scale research project-ni bishoyot discussion khorok.",
        "expected_entities": [
            ("10 June 2027", "MISC"),
            ("Agartala", "LOC"),
            ("Tripura University", "ORG"),
            ("Professor Debashis Jamatia", "PER"),
            ("Dr. Rina Debbarma", "PER"),
            ("Dr. Anupam Chakma", "PER"),
            ("Professor Michael Das", "PER"),
            ("Suman Reang", "PER"),
            ("Priya Jamatia", "PER"),
            ("Tripura", "LOC"),
            ("Assam", "LOC"),
            ("Mizoram", "LOC"),
            ("Manipur", "LOC"),
            ("Meghalaya", "LOC"),
            ("Bangladesh", "LOC"),
            ("New Delhi", "LOC"),
            ("AI4Bharat", "ORG"),
            ("Google Research", "ORG"),
            ("Microsoft Research", "ORG"),
            ("University of Dhaka", "ORG"),
            ("Indian Institute of Technology Madras", "ORG"),
            ("Kokborok", "MISC"),
            ("Assamese", "MISC"),
            ("Bengali", "MISC"),
            ("Hindi", "MISC"),
            ("Tamil", "MISC"),
            ("Telugu", "MISC"),
            ("Named Entity Recognition", "MISC"),
            ("Machine Translation", "MISC"),
            ("Attention Attribution", "MISC"),
            ("Explainable Artificial Intelligence", "MISC"),
        ]
    },
]


def clean_text(t: str) -> str:
    # Strip invisible Unicode joiners (ZWJ, ZWNJ)
    t = t.replace('\u200d', '').replace('\u200c', '')
    # Strip Indic and Kokborok case clitics for fuzzy matching
    t = re.sub(r'-(?:ao|ot|ni|o|bai|rok|rog|no|le|de|bo|se)$', '', t.strip())
    t = re.sub(r'^(?:Dr\.?|Prof\.?|Professor|Mr\.?|Mrs\.?|Ms\.|ড°|অধ্যাপক|আচাৰ্য|මහාචාර්ය|ආචාර්ය|प्रा\.|डॉ\.)\s+', '', t, flags=re.I)
    t = re.sub(r'(?:යේදී|දී)$', '', t.strip())
    t = t.rstrip(',.;: ')
    return t.strip().lower()




def evaluate_20():
    print("=" * 80)
    print("20-SENTENCE MULTILINGUAL NER BENCHMARK: 4 LANGUAGES x 5 SENTENCES")
    print("=" * 80)

    all_results = []
    total_expected_all = 0
    total_matched_all = 0

    # Group by language to reuse predictor bundles
    from itertools import groupby
    grouped = {}
    for item in DATASET_20:
        grouped.setdefault(item["language"], []).append(item)

    for lang, items in grouped.items():
        print(f"\n{'='*30} LANGUAGE: {lang.upper()} {'='*30}")
        predictor = get_predictor(lang)
        lang_expected = 0
        lang_matched = 0

        for item in items:
            sid = item["id"]
            title = item["title"]
            sent = item["sentence"]
            expected = item["expected_entities"]
            lang_expected += len(expected)

            print(f"\n[{sid}] {title}")
            print(f"  * Expected: {len(expected)} entities")

            res = predictor.predict(sent, include_attribution=False)
            pred_entities = res["entities"]
            print(f"  * Model Extracted: {len(pred_entities)} entities")

            matched = []
            missed = []

            for exp_text, exp_type in expected:
                clean_exp = clean_text(exp_text)
                found = False
                for p in pred_entities:
                    clean_pred = clean_text(p["text"])
                    if (clean_exp == clean_pred or clean_exp in clean_pred or clean_pred in clean_exp) and exp_type == p["type"]:
                        matched.append((exp_text, exp_type, p["text"]))
                        found = True
                        break
                if not found:
                    missed.append((exp_text, exp_type))

            lang_matched += len(matched)
            recall = (len(matched) / len(expected) * 100) if expected else 0.0
            prec = (len(matched) / len(pred_entities) * 100) if pred_entities else 0.0

            print(f"  * Matched: {len(matched)} / {len(expected)} ({recall:.1f}% Recall | {prec:.1f}% Precision)")
            if missed:
                print(f"  * Missed Entities ({len(missed)}): {[(m[0], m[1]) for m in missed]}")
            else:
                print("  * Missed: 0 (FLAWLESS 100% RECOVERY)")

            all_results.append({
                "language": lang,
                "id": sid,
                "title": title,
                "sentence": sent,
                "expected_count": len(expected),
                "predicted_count": len(pred_entities),
                "matched_count": len(matched),
                "missed": missed,
                "recall": round(recall, 1),
                "precision": round(prec, 1),
                "predicted_entities": pred_entities,
            })

        lang_rec = (lang_matched / lang_expected * 100) if lang_expected else 0.0
        print(f"\n--> {lang.upper()} SUB-TOTAL: {lang_matched} / {lang_expected} ({lang_rec:.1f}% Recall)")
        total_expected_all += lang_expected
        total_matched_all += lang_matched

    overall_recall = (total_matched_all / total_expected_all * 100) if total_expected_all else 0.0
    print("\n" + "=" * 80)
    print(f"GRAND TOTAL ACROSS ALL 4 LANGUAGES: {total_matched_all} / {total_expected_all} ({overall_recall:.1f}% Recall)")
    print("=" * 80)

    out_file = PROJECT_ROOT / "reports" / "20_multilingual_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[REPORT SAVED] Results written to {out_file}")


if __name__ == "__main__":
    evaluate_20()
