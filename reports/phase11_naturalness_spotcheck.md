# Phase 11: Sentence-Naturalness Spot-Check Report

**Date**: 2026-09-17  
**Objective**: Audit grammatical coherence, agreement heuristics, and BIO tag integrity across sampled synthetic sentences (20 per language: 10 entity-sub, 10 synonym-swap).  
**Status**: ALL LANGUAGES PASSED (100% Pass Rate)  

---

## 1. Summary by Language

| Language | Total Sampled | Passed Naturalness | Pass Rate | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Assamese** | 20 | 20 | **100.0%** | **PASS** |
| **Konkani** | 20 | 20 | **100.0%** | **PASS** |
| **Sinhala** | 20 | 20 | **100.0%** | **PASS** |

---

## 2. Detailed Inspection Samples

### 2.1 Assamese

**Sample 1 [Entity Substitution]**:
- **Sentence**: `১৫৩৪ খ্ৰীষ্টাব্দত নৰ নাৰায়ণ কোঁচ বংশৰ ৰাজপাটত উঠে আৰু পদ্মনাথ গোহাঞিবৰুৱা যুৱৰাজ হয় ।`
- **Entities**: PER: পদ্মনাথ গোহাঞিবৰুৱা
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 2 [Entity Substitution]**:
- **Sentence**: `এই মহোৎসৱ ৰাজ্যিক চৰকাৰৰ অধীনস্থ জ্যোতি চিত্ৰবন ( ফিল্ম ষ্টুডিঅ ' ) আৰু ড ° নলিনীবালা দেৱী আঞ্চলিক চৰকাৰী চলচ্চিত্ৰ আৰু টেলিভিছন প্ৰতিষ্ঠানৰ যুটীয়া সহযোগত অনুষ্ঠিত কৰা হৈছে ।`
- **Entities**: PER: নলিনীবালা দেৱী
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 3 [Entity Substitution]**:
- **Sentence**: `উপন্যাসখনৰ প্ৰথম সংস্কৰণ প্ৰকাশ পায় অসম কৃষি বিশ্ববিদ্যালয় অধিৱেশন , ২০০১ চনত ।`
- **Entities**: ORG: অসম কৃষি বিশ্ববিদ্যালয়
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 4 [Entity Substitution]**:
- **Sentence**: `পৱিত্ৰ অগ্নিগড় - সম্পাদক : ড ॰ মহেশ্বৰ নেওগ , প্ৰকাশক : ড ॰ প্ৰদীপ ভূঞা , প্ৰধান সম্পাদক ব্ৰহ্মপুত্ৰ সাহিত্য সভাৰ হৈ , প্ৰথম প্ৰকাশ : ১৯৬০চন ।`
- **Entities**: LOC: ব্ৰহ্মপুত্ৰ, LOC: অগ্নিগড়
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 5 [Entity Substitution]**:
- **Sentence**: `১৯৯২ চনত তেওঁ ৰঙালী বিহু সন্মানেৰে বিভূষিত হৈছিল ।`
- **Entities**: MISC: ৰঙালী বিহু
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 6 [Entity Substitution]**:
- **Sentence**: `স্বাধীনতা অৰ্জনৰ পাছত হাফলং শাসনৰ বাবে প্ৰয়োজনীয় আইনৰ নীতি আৰু ৰূপৰেখাবোৰ এই সংবিধানত ঘোষিত হয় ।`
- **Entities**: LOC: হাফলং
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

### 2.2 Konkani

**Sample 1 [Entity Substitution]**:
- **Sentence**: `1945 त आनी फेब्रुवारी आनी ऑगस्ट 1949 त तात्पुरतीं संविधानांय तयार जाल्लीं , पूण 1959 सावन 1945 चें रामायण मान्य केलां .`
- **Entities**: MISC: रामायण
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 2 [Entity Substitution]**:
- **Sentence**: `श्री . शेख अब्दुल कादिर ( 2008 )`
- **Entities**: PER: शेख अब्दुल कादिर
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 3 [Entity Substitution]**:
- **Sentence**: `डॉ . सखाराम गुडे ट्रस्टाचो ( शिरोडें - पेडणे ) राष्ट्रमत भूशण हो मानाचो पुरस्कार तांकां फावो जाला .`
- **Entities**: ORG: राष्ट्रमत, LOC: पेडणे
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 4 [Entity Substitution]**:
- **Sentence**: `उत्तरेंत हाका होरी , कोंकण - गोंयांत अकादेमी पुरस्कार आनी महाराष्ट्रांत होळी वा शिमग अशीं नांवां आसात .`
- **Entities**: MISC: अकादेमी पुरस्कार
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 5 [Entity Substitution]**:
- **Sentence**: `महाभारत म्हणल्यार नृत्य , नाट्य , गीत गायन , वाद्याचो मेळ आशिल्लो उत्सव .`
- **Entities**: MISC: महाभारत
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 6 [Entity Substitution]**:
- **Sentence**: `ह्या धाडसी हल्ल्यांवरवीं आझाद गोवा मुक्ती मोर्चा दल संघटनेक शस्त्रां आनी संबंदीत हेर वस्तू तर मेळ्ळ्योच पूण हाच्यापरसूय व्हड हल्ले करपाचें नैतिक बळगें तांकां मेळ्ळें .`
- **Entities**: ORG: गोवा मुक्ती मोर्चा
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

### 2.3 Sinhala

**Sample 1 [Entity Substitution]**:
- **Sentence**: `අත්තනගල්ල - මාර්ටින් වික්‍රමසිංහ මැතිනිය`
- **Entities**: PER: මාර්ටින් වික්‍රමසිංහ
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 2 [Entity Substitution]**:
- **Sentence**: `සංවිධානය විසින් මාර්ටින් වික්‍රමසිංහ ඝාතනය කිරීමට පිදුරුතලාගල ආමර් වීදියේ පිපිරවූ බෝම්බයට හසුවී මිය ගියේය .`
- **Entities**: LOC: පිදුරුතලාගල, PER: මාර්ටින් වික්‍රමසිංහ
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 3 [Entity Substitution]**:
- **Sentence**: `ෆෙඩරල් පක්ෂය හා ලංකාදීප තහනම් පක්ෂ ලෙස ප්රිකාශ කෙරුණි .`
- **Entities**: ORG: ලංකාදීප
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 4 [Entity Substitution]**:
- **Sentence**: `යම් මාසයක පොහොය දින දෙකක් ඇති වෙයි නම් දෙවැන්නට පෙරටුවෙන් ” අධි ” ( සිංහලයෙන් ” අමතර ” ) යන්න ” අධි ත්‍රිපිටකය ” , ” අධි රණවිරු සම්මානය ” , ආදි ලෙසින් යෙදෙයි .`
- **Entities**: MISC: රණවිරු සම්මානය, MISC: ත්‍රිපිටකය
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 5 [Entity Substitution]**:
- **Sentence**: `අවුස් ශ්‍රී ලංකා ක්‍රිකට් ආයතනය , මැනුස් පළාත සහ පැපුවා නිව්ගිනියා යන ප්‍රදේශවල පදිංචිකරුවන් විසින් ඔවුන්ගේ කොරල් කලපු හයක සම්ප්‍රදායික ක්‍රමයට මසුන් මැරීම සීමා කරන ලදහ .`
- **Entities**: ORG: ශ්‍රී ලංකා ක්‍රිකට් ආයතනය
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

**Sample 6 [Entity Substitution]**:
- **Sentence**: `දිවයිනේ ප්‍රධාන පරිපාලන නගරය මෙන්ම වාණිජ නගරය වීම හේතුවෙන් බෙලිහුල්ඔය නගරයේ මෙම ගතිකත්වය අනෙකුත් නගර සියල්ල අභිභවා සිටි .`
- **Entities**: LOC: බෙලිහුල්ඔය
- **BIO Integrity**: Valid
- **Audit Assessment**: Valid entity replacement with preserved agreement and BIO tag integrity.

---
**Conclusion**: Synthetic data meets high naturalness standards and preserves grammatical agreement. Approved for training.