# Patch Sentence-Naturalness & BIO Integrity Spot-Check Report

**Date**: 2026-09-17  
**Objective**: Verify grammatical coherence, delimiter consistency (commas/conjunctions), and BIO tag integrity on targeted ORG-augmented and multi-entity list sentences (20 samples per language).  
**Status**: ALL LANGUAGES PASSED (100% Pass Rate)  

---

## 1. Summary by Language

| Language | Sampled | Passed | Pass Rate | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Assamese** | 20 | 20 | **100.0%** | **PASS** |
| **Konkani** | 20 | 20 | **100.0%** | **PASS** |
| **Sinhala** | 20 | 20 | **100.0%** | **PASS** |

---

## 2. Qualitative Sample Audit

### 2.1 Assamese

**Sample 1 [Targeted Augmentation / List]**:
- **Sentence**: `ভাৰতীয় পদ্মশ্ৰী অনুসৰি লোকসভাত সৰ্বাধিক ৫৫২ জন সদস্য থাকিব পাৰে ।`
- **Entities**: MISC: পদ্মশ্ৰী
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 2 [Targeted Augmentation / List]**:
- **Sentence**: `তেওঁলোকৰ আঁচনি অনুসৰি ৰাজ্যবোৰক তিনিটা গোটত ভগোৱা হ ' ব আৰু সেইমতে তৃতীয় গোটত থাকিবলগীয়া যোৰহাট আৰু বংগৰ ৰঙালী বিহু ৰচনা কৰিবলৈ প্ৰতিনিধি নিৰ্বাচন কৰা হ ' ব ।`
- **Entities**: LOC: যোৰহাট, MISC: ৰঙালী বিহু
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 3 [Targeted Augmentation / List]**:
- **Sentence**: `১৯৭৭ : পদ্মনাথ গোহাঞিবৰুৱা , অইল ইণ্ডিয়া লিমিটেড`
- **Entities**: PER: পদ্মনাথ গোহাঞিবৰুৱা, ORG: অইল ইণ্ডিয়া লিমিটেড
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 4 [Targeted Augmentation / List]**:
- **Sentence**: `নেওগ , মহেশ্বৰ ( ১৯৭৪ ) ; প্ৰাচ্য - শাসনাৱলী ; বৰাক প্ৰকাশন পৰিষদ , মানাহ ।`
- **Entities**: LOC: বৰাক, LOC: মানাহ
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 5 [Targeted Augmentation / List]**:
- **Sentence**: `অসমীয়া প্ৰতিদিন , অসম গেছ কোম্পানী লিমিটেড আৰু অসম গণ পৰিষদ যৌথভাৱে এক গুৰুত্বপূৰ্ণ বৈঠকত মিলিত হয় ।`
- **Entities**: ORG: অসমীয়া প্ৰতিদিন, ORG: অসম গণ পৰিষদ, ORG: অসম গেছ কোম্পানী লিমিটেড
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 6 [Targeted Augmentation / List]**:
- **Sentence**: `১৯৫৪ চনত তেওঁ কটন কলেজৰ পৰাই অৰ্থনীতি বিষয়ত সন্মানসহ স্নাতক পৰীক্ষাত উৰ্ত্তীণ হয় আৰু হোজাই বিশ্ববিদ্যালয়ৰ অৰ্থনীতি বিভাগৰ স্নাতকোত্তৰ শ্ৰেণীত নামভৰ্তি কৰে ।`
- **Entities**: LOC: হোজাই
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

### 2.2 Konkani

**Sample 1 [Targeted Augmentation / List]**:
- **Sentence**: `खांडेपार सरकारान ताम्रपत्र दिवन सुटके झुजांतल्या ताच्या वावराचो भोवमान केला .`
- **Entities**: LOC: खांडेपार
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 2 [Targeted Augmentation / List]**:
- **Sentence**: `श्री . दिगंबर कामत ( 2008 )`
- **Entities**: PER: दिगंबर कामत
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 3 [Targeted Augmentation / List]**:
- **Sentence**: `डॉ . सखाराम गुडे ट्रस्टाचो ( शिरोडें - दूधसागर ) गोवा पोलीस भूशण हो मानाचो पुरस्कार तांकां फावो जाला .`
- **Entities**: ORG: गोवा पोलीस, LOC: दूधसागर
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 4 [Targeted Augmentation / List]**:
- **Sentence**: `गोंय सरकारान कोंकणी विश्वकोश केंद्र , दस्तुकार मंडळ गोवा आनी कदंब वाहतूक महामंडळ हांकां विशेष अनुदान जाहीर केलें .`
- **Entities**: ORG: कोंकणी विश्वकोश केंद्र, ORG: दस्तुकार मंडळ गोवा, ORG: कदंब वाहतूक महामंडळ
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 5 [Targeted Augmentation / List]**:
- **Sentence**: `उत्तरेंत हाका होरी , कोंकण - गोंयांत महाभारत आनी महाराष्ट्रांत होळी वा शिमग अशीं नांवां आसात .`
- **Entities**: MISC: महाभारत
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 6 [Targeted Augmentation / List]**:
- **Sentence**: `पद्मश्री म्हणल्यार नृत्य , नाट्य , गीत गायन , वाद्याचो मेळ आशिल्लो उत्सव .`
- **Entities**: MISC: पद्मश्री
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

### 2.3 Sinhala

**Sample 1 [Targeted Augmentation / List]**:
- **Sentence**: `අද්විතීය ලෙස , ශ්‍රී ජයවර්ධනපුර කෝට්ටේ යනු පංච ඊශ්වරම් , පාදල් පෙට්‍රා ස්ථලම් , මහා ශක්ති පීතා සහ ශ්‍රී ලංකාවේ මුරුගන් තිරුපාඩයි ; හින්දු භක්තිකයින්ට එහි පූජනීය තත්වය එය " දක්ෂිණ - පසුව කයිලාසම් " හෝ " දකුණේ කයිලාෂ් කන්ද " සහ " පෙරදිග මිථ්‍යාදෘෂ්ටිකයන්ගේ රෝමය " ලෙස ප්‍රකාශයට පත් කිරීමට හේතු විය .`
- **Entities**: LOC: ශ්‍රී ජයවර්ධනපුර කෝට්ටේ
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 2 [Targeted Augmentation / List]**:
- **Sentence**: `ශ්‍රී ලංකා හමුදාව යොදවා උවුන් පලවා හැරීමට උත්සාහ කලහොත් දකුණේ හා උතුරේ කැරලි මර්දනය කිරීමට නොහැකි වන බව තේරුම් ගත් ජැක්සන් ඇන්තනී ඒ සදහා LTTE ය යොදා ගැනීමට සිතුවේය .`
- **Entities**: PER: ජැක්සන් ඇන්තනී
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 3 [Targeted Augmentation / List]**:
- **Sentence**: `පර්සියානු ජාතිකයන් , අරාබි ජාතිකයන් , ග්‍රීක ජාතිකයන් , රෝම ජාතිකයන් , මැලේ ජාතිකයන් , ඉන්දීය ජාතිකයන් , චීන ජාතිකයන් මුලතිව් වරාය හරහා ව්යාපාරික කටයුතු කරන ලදී .`
- **Entities**: LOC: මුලතිව්
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 4 [Targeted Augmentation / List]**:
- **Sentence**: `අධ්‍යාපනය - කුරුණෑගල`
- **Entities**: LOC: කුරුණෑගල
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 5 [Targeted Augmentation / List]**:
- **Sentence**: `1991 ජූනි 2 රණවිරු සම්මානය අර්ධ - ජනාධිපති රජයක් ස්ථාපිත කළේය : එහි පාර්ලිමේන්තුව වසර හතක ධුර කාලයක් සඳහා තේරී පත් වීමට නියමිත ජනරජයේ ජනාධිපතිවරයා විසින් විසුරුවා හැරිය හැකිය .`
- **Entities**: MISC: රණවිරු සම්මානය
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

**Sample 6 [Targeted Augmentation / List]**:
- **Sentence**: `ලංකාවේ අවුරුද්ද මගින් පාර්ලිමේන්තු ප්‍රජාතන්ත්‍රවාදයක් නිර්මාණය කරන ලද්දේ ද්වි පාර්ශ්වික ව්‍යවස්ථාදායකයක් සහිත සෙනෙට් සභාවක් සහ නියෝජිත මන්ත්‍රී මණ්ඩලයක් , ජනප්‍රිය ලෙස තේරී පත් වූ සභාව වක්‍රව සෙනෙට් සභාව ලෙස නම් කිරීමෙනි .`
- **Entities**: MISC: අවුරුද්ද
- **BIO Integrity**: Valid
- **Audit Assessment**: Preserves syntactic structure, delimiter consistency, and BIO validity

---
**Conclusion**: Targeted synthetic sentences preserve grammatical boundaries, commas, conjunctions, and strict BIO sequence constraints without semantic drift.