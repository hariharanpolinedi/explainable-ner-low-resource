# Reliability Cycle 2 — Step 2: Assamese Training Label Correction Audit Log

> **Deliverable**: Traceable, guideline-referenced label corrections for Assamese NER.
> **Input Corpus**: `data\gold\assamese\train.conll` (850 sentences, 14,649 tokens).
> **Output Corpus**: `data\gold\assamese\train_cycle2_corrected.conll` (Preserving original unmodified).
> **Total Corrections Applied**: **699** tokens (4.77% of training corpus).

---

## 1. Executive Summary & Policy Compliance

Per the Cycle 2 Step 2 technical decision rules:
1. **Zero Blanket Majority-Vote Relabeling**: Every correction was audited against explicit sentential context. Generic uses of nouns (`সভা`, `চৰকাৰ`, `পাৰ্টি`, `বিশ্ববিদ্যালয়`, `বঁটা`) were strictly preserved as `O`.
2. **Direct Traceability to Guidelines**: Every modified label directly cites `annotation/guidelines_assamese.md`.
3. **Root Cause Rectification**: Remediated the massive false-negative label noise identified in Step 1b (where canonical figures like `ভূপেন হাজৰিকা`, `লক্ষ্মীনাথ বেজবৰুৱা`, `শ্ৰীমন্ত শংকৰদেৱ`, and inflected locations like `ভাৰতৰ`, `অসমৰ` had been mislabeled as `O` up to 85.9% of the time).

## 2. Correction Breakdown by Entity Type

| Entity Tag | Number of Corrections | % of All Corrections | Representative Corrected Tokens |
| :--- | :---: | :---: | :--- |
| **`I-ORG`** | 339 | 48.5% | `সাহিত্য`, `সভাৰ`, `চৰকাৰৰ`, `বিশ্ববিদ্যালয়ৰ` |
| **`B-LOC`** | 99 | 14.2% | `অসমৰ`, `ভাৰতৰ`, `গুৱাহাটীত`, `অসমত` |
| **`B-PER`** | 98 | 14.0% | `ভূপেন`, `শ্ৰীমন্ত`, `লক্ষ্মীনাথ`, `হেমচন্দ্ৰ` |
| **`I-PER`** | 94 | 13.4% | `হাজৰিকাই`, `হাজৰিকাৰ`, `বেজবৰুৱাৰ`, `কুমাৰ` |
| **`B-ORG`** | 49 | 7.0% | `ভাৰতীয়`, `সদৌ`, `কটন` |
| **`I-MISC`** | 12 | 1.7% | `আইৰ`, `বিহুত`, `সাধুৰ`, `বিহুৰ` |
| **`B-MISC`** | 8 | 1.1% | `বুঢ়ী`, `ভোগালী`, `ৰঙালী`, `জ্ঞানপীঠ` |

---

## 3. Ranked Table of Corrected Tokens

| Rank | Token | Corrections Count | Primary Target Tag | Annotation Guideline Section | Diagnostic Finding Traced To |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | **`সাহিত্য`** | 63 | `I-ORG` | Guidelines Section 2.3 & 3 | Audit Rank / Step 1b Finding |
| 2 | **`সভাৰ`** | 55 | `I-ORG` | Guidelines Section 2.3 & 3 | Audit Rank / Step 1b Finding |
| 3 | **`ভাৰতীয়`** | 40 | `B-ORG` | Guidelines Section 2.3 | Audit Rank / Step 1b Finding |
| 4 | **`অসমৰ`** | 38 | `B-LOC` | Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC | Audit Rank / Step 1b Finding |
| 5 | **`চৰকাৰৰ`** | 35 | `I-ORG` | Guidelines Section 3, Example 1 | Audit Rank / Step 1b Finding |
| 6 | **`বিশ্ববিদ্যালয়ৰ`** | 33 | `I-ORG` | Guidelines Section 2.3 | Audit Rank / Step 1b Finding |
| 7 | **`চৰকাৰে`** | 29 | `I-ORG` | Guidelines Section 3, Example 1 | Audit Rank / Step 1b Finding |
| 8 | **`জাতীয়`** | 28 | `I-ORG` | Guidelines Section 2.3 | Audit Rank / Step 1b Finding |
| 9 | **`ভাৰতৰ`** | 24 | `B-LOC` | Guidelines Section 2.2 & Section 3 | Audit Rank / Step 1b Finding |
| 10 | **`কংগ্ৰেছৰ`** | 22 | `I-ORG` | Guidelines Section 2.3 | Audit Rank / Step 1b Finding |
| 11 | **`ভূপেন`** | 19 | `B-PER` | Guidelines Section 2.1 | Audit Rank / Step 1b Finding |
| 12 | **`শ্ৰীমন্ত`** | 10 | `B-PER` | Guidelines Section 2.1 | Audit Rank / Step 1b Finding |
| 13 | **`লক্ষ্মীনাথ`** | 10 | `B-PER` | Guidelines Section 2.1 | Audit Rank / Step 1b Finding |
| 14 | **`জনতা`** | 10 | `I-ORG` | Guidelines Section 2.3 | Audit Rank / Step 1b Finding |
| 15 | **`হাজৰিকাই`** | 9 | `I-PER` | Guidelines Section 2.1 | Audit Rank / Step 1b Finding |
| 16 | **`হেমচন্দ্ৰ`** | 8 | `B-PER` | Guidelines Section 2.1 | Audit Rank / Step 1b Finding |
| 17 | **`জ্যোতিপ্ৰসাদ`** | 8 | `B-PER` | Guidelines Section 2.1 | Audit Rank / Step 1b Finding |
| 18 | **`বিধানসভাৰ`** | 7 | `I-ORG` | Guidelines Section 2.3 | Audit Rank / Step 1b Finding |
| 19 | **`হাজৰিকাৰ`** | 7 | `I-PER` | Guidelines Section 2.1 | Audit Rank / Step 1b Finding |
| 20 | **`প্ৰফুল্ল`** | 6 | `B-PER` | Guidelines Section 2.1 | Audit Rank / Step 1b Finding |
| 21 | **`বেজবৰুৱাৰ`** | 6 | `I-PER` | Guidelines Section 2.1 | Audit Rank / Step 1b Finding |
| 22 | **`তৰুণ`** | 6 | `B-PER` | Guidelines Section 2.1 | Audit Rank / Step 1b Finding |
| 23 | **`গুৱাহাটীত`** | 5 | `B-LOC` | Guidelines Section 2.2 & Section 3 | Audit Rank / Step 1b Finding |
| 24 | **`অসমত`** | 5 | `B-LOC` | Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC | Audit Rank / Step 1b Finding |
| 25 | **`গুৱাহাটীৰ`** | 5 | `B-LOC` | Guidelines Section 2.2 & Section 3 | Audit Rank / Step 1b Finding |

---

## 4. Complete Audit Trail (Sample First 50 Corrections)

| Sent ID | Token | Pos | Change | Sentential Context | Guideline Justification |
| :---: | :--- | :---: | :---: | :--- | :--- |
| #5 | `ভাৰতৰ` | 13 | `O` $\to$ **`B-LOC`** | "...১৯৬২ চনত ভাৰতৰ ৰাষ্ট্ৰপতিৰ পৰা পদ্মভূষণ..." | Inflected location noun for India (ভাৰতৰ) (Guidelines Section 2.2 & Section 3) |
| #5 | `চৰকাৰৰ` | 22 | `O` $\to$ **`I-ORG`** | "...চনত অসম চৰকাৰৰ শ্ৰীমন্ত শংকৰদেৱ..." | Government as political administration institution (Guidelines Section 3, Example 1) |
| #6 | `বিশ্ববিদ্যালয়ৰ` | 2 | `O` $\to$ **`I-ORG`** | "...তেখেতে গুৱাহাটী বিশ্ববিদ্যালয়ৰ উপাচাৰ্য হিচাপেও..." | University institutional body following named institution stem (Guidelines Section 2.3) |
| #9 | `চৰকাৰে` | 3 | `O` $\to$ **`I-ORG`** | "...চনত অসম চৰকাৰে কটন কলেজ..." | Government as political administration institution (Guidelines Section 3, Example 1) |
| #9 | `কটন` | 4 | `O` $\to$ **`B-ORG`** | "...অসম চৰকাৰে কটন কলেজ ৰাজ্যিক..." | Cotton College/University institutional entity (Guidelines Section 2.3) |
| #9 | `কটন` | 15 | `O` $\to$ **`B-ORG`** | "...ৰাখিলে তথা কটন কলেজক এখন..." | Cotton College/University institutional entity (Guidelines Section 2.3) |
| #10 | `বিশ্ববিদ্যালয়ে` | 4 | `O` $\to$ **`I-ORG`** | "...আৰু ডিব্ৰুগড় বিশ্ববিদ্যালয়ে তেখেতক ডি..." | University institutional body following named institution stem (Guidelines Section 2.3) |
| #13 | `বিশ্ববিদ্যালয়ৰ` | 13 | `O` $\to$ **`I-ORG`** | "...ই ডিব্ৰুগড় বিশ্ববিদ্যালয়ৰ অধীনলৈ আহে..." | University institutional body following named institution stem (Guidelines Section 2.3) |
| #14 | `চুক্তিমতে` | 6 | `O` $\to$ **`I-MISC`** | "...গোচৰো অসম চুক্তিমতে ১৯৮৫ চনত..." | Continuation of Assam Accord historical treaty (Guidelines Section 2.4) |
| #28 | `হেমচন্দ্ৰ` | 4 | `O` $\to$ **`B-PER`** | "...সভা আৰু হেমচন্দ্ৰ গোস্বামী..." | Proper personal name of Hemchandra Barua / Goswami (Guidelines Section 2.1) |
| #28 | `গোস্বামী` | 5 | `O` $\to$ **`I-PER`** | "...আৰু হেমচন্দ্ৰ গোস্বামী..." | Surname of Hemchandra Barua/Goswami (Guidelines Section 2.1) |
| #31 | `প্ৰফুল্ল` | 2 | `O` $\to$ **`B-PER`** | "...লগে লগে প্ৰফুল্ল কুমাৰ মহন্ত নেতৃত্বাধীন..." | Proper personal name of Prafulla Kumar Mahanta (Guidelines Section 2.1) |
| #31 | `কুমাৰ` | 3 | `O` $\to$ **`I-PER`** | "...লগে প্ৰফুল্ল কুমাৰ মহন্ত নেতৃত্বাধীন..." | Middle name of Prafulla Kumar Mahanta (Guidelines Section 2.1) |
| #31 | `মহন্ত` | 4 | `O` $\to$ **`I-PER`** | "...প্ৰফুল্ল কুমাৰ মহন্ত নেতৃত্বাধীন প্ৰাক্তন..." | Surname of Prafulla Kumar Mahanta (Guidelines Section 2.1) |
| #39 | `অসমৰ` | 3 | `O` $\to$ **`B-LOC`** | "...বিশ্ববিদ্যালয় উজনি অসমৰ প্ৰথম বিশ্ববিদ্যালয় ।..." | Inflected location noun for Assam (অসমৰ) (Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC) |
| #44 | `প্ৰফুল্ল` | 2 | `O` $\to$ **`B-PER`** | "...তাৰ পিছতে প্ৰফুল্ল কুমাৰ মহন্তৰ নেতৃত্বত..." | Proper personal name of Prafulla Kumar Mahanta (Guidelines Section 2.1) |
| #44 | `কুমাৰ` | 3 | `O` $\to$ **`I-PER`** | "...পিছতে প্ৰফুল্ল কুমাৰ মহন্তৰ নেতৃত্বত..." | Middle name of Prafulla Kumar Mahanta (Guidelines Section 2.1) |
| #44 | `মহন্তৰ` | 4 | `O` $\to$ **`I-PER`** | "...প্ৰফুল্ল কুমাৰ মহন্তৰ নেতৃত্বত অসম..." | Surname of Prafulla Kumar Mahanta (Guidelines Section 2.1) |
| #45 | `যোৰহাটৰ` | 16 | `O` $\to$ **`B-LOC`** | "...প্ৰতিষ্ঠান আদি যোৰহাটৰ অন্যতম উল্লেখযোগ্য প্ৰতিষ্ঠান..." | Inflected city location noun (যোৰহাটৰ) (Guidelines Section 2.2 & Section 3) |
| #47 | `ভাৰতৰ` | 4 | `O` $\to$ **`B-LOC`** | "...ছাত্ৰ সন্থা ভাৰতৰ স্বাধীনতা আন্দোলনৰ অনুৰূপে..." | Inflected location noun for India (ভাৰতৰ) (Guidelines Section 2.2 & Section 3) |
| #48 | `অসমৰ` | 5 | `O` $\to$ **`B-LOC`** | "...) হৈছে অসমৰ এখন পুৰণি অসমীয়া..." | Inflected location noun for Assam (অসমৰ) (Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC) |
| #52 | `চৰকাৰৰ` | 9 | `O` $\to$ **`I-ORG`** | "...চনত অসম চৰকাৰৰ সাংস্কৃতিক সঞ্চালকালয়ৰ..." | Government as political administration institution (Guidelines Section 3, Example 1) |
| #53 | `হেমচন্দ্ৰ` | 3 | `O` $\to$ **`B-PER`** | "...বেজবৰুৱা আৰু হেমচন্দ্ৰ গোস্বামীৰ লগ লাগি..." | Proper personal name of Hemchandra Barua / Goswami (Guidelines Section 2.1) |
| #53 | `গোস্বামীৰ` | 4 | `O` $\to$ **`I-PER`** | "...আৰু হেমচন্দ্ৰ গোস্বামীৰ লগ লাগি..." | Surname of Hemchandra Barua/Goswami (Guidelines Section 2.1) |
| #55 | `অসমৰ` | 11 | `O` $\to$ **`B-LOC`** | "...২০০৩ ) অসমৰ এগৰাকী ঔপন্যাসিক ,..." | Inflected location noun for Assam (অসমৰ) (Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC) |
| #60 | `শ্ৰীমন্ত` | 3 | `O` $\to$ **`B-PER`** | "...কথা : শ্ৰীমন্ত শঙ্কৰদেৱ আৰু শ্ৰীশ্ৰী..." | Proper personal name of Srimanta Sankardev (Guidelines Section 2.1) |
| #60 | `শঙ্কৰদেৱ` | 4 | `O` $\to$ **`I-PER`** | "...: শ্ৰীমন্ত শঙ্কৰদেৱ আৰু শ্ৰীশ্ৰী..." | Surname/continuation of Srimanta Sankardev (Guidelines Section 2.1) |
| #62 | `অসমৰ` | 5 | `O` $\to$ **`B-LOC`** | "...গগৈ , অসমৰ প্ৰাক্তন মুখ্যমন্ত্ৰী ।..." | Inflected location noun for Assam (অসমৰ) (Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC) |
| #63 | `গুৱাহাটীত` | 0 | `O` $\to$ **`B-LOC`** | "...গুৱাহাটীত অৱস্থিত লোকপ্ৰিয় গোপীনাথ..." | Inflected city location noun (গুৱাহাটীত) (Guidelines Section 2.2 & Section 3) |
| #68 | `অসমৰ` | 5 | `O` $\to$ **`B-LOC`** | "...বিশ্ববিদ্যালয় , অসমৰ নগাঁওত অৱস্থিত এখন..." | Inflected location noun for Assam (অসমৰ) (Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC) |
| #70 | `ভাৰতৰ` | 9 | `O` $\to$ **`B-LOC`** | "...১৯৩৯ ) ভাৰতৰ স্বাধীনতা আন্দোলনৰ এগৰাকী..." | Inflected location noun for India (ভাৰতৰ) (Guidelines Section 2.2 & Section 3) |
| #70 | `অসমৰ` | 13 | `O` $\to$ **`B-LOC`** | "...আন্দোলনৰ এগৰাকী অসমৰ নেতা আছিল ।..." | Inflected location noun for Assam (অসমৰ) (Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC) |
| #73 | `গুৱাহাটীত` | 4 | `O` $\to$ **`B-LOC`** | "...কলাক্ষেত্ৰ , গুৱাহাটীত অৱস্থিত অসমীয়া কলা..." | Inflected city location noun (গুৱাহাটীত) (Guidelines Section 2.2 & Section 3) |
| #83 | `বিষ্ণুপ্ৰসাদ` | 3 | `O` $\to$ **`B-PER`** | "...আগৰৱালা আৰু বিষ্ণুপ্ৰসাদ ৰাভাৰ সান্নিধ্য লাভ..." | Proper personal name of Bishnuprasad Rabha (Guidelines Section 2.1) |
| #83 | `ৰাভাৰ` | 4 | `O` $\to$ **`I-PER`** | "...আৰু বিষ্ণুপ্ৰসাদ ৰাভাৰ সান্নিধ্য লাভ..." | Surname of Bishnuprasad Rabha (Guidelines Section 2.1) |
| #83 | `জয়মতী` | 14 | `O` $\to$ **`B-PER`** | "...কথাছবি " জয়মতী " ত অভিনয়..." | Historical personal name of Joymoti Kunwori (Guidelines Section 2.1) |
| #84 | `অসমত` | 13 | `O` $\to$ **`B-LOC`** | "...পৰে , অসমত থকা সময়ত তেওঁ..." | Inflected location noun for Assam (অসমত) (Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC) |
| #84 | `জুবিন` | 20 | `O` $\to$ **`B-PER`** | "...হাজৰিকা , জুবিন গাৰ্গ , জিতুল..." | Proper personal name of Zubeen Garg (Guidelines Section 2.1) |
| #84 | `গাৰ্গ` | 21 | `O` $\to$ **`I-PER`** | "..., জুবিন গাৰ্গ , জিতুল..." | Surname of Zubeen Garg (Guidelines Section 2.1) |
| #87 | `অসমত` | 4 | `O` $\to$ **`B-LOC`** | "...মাঘী পূৰ্ণিমা অসমত বীৰ চিলাৰায় দিৱস..." | Inflected location noun for Assam (অসমত) (Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC) |
| #88 | `অসমৰ` | 6 | `O` $\to$ **`B-LOC`** | "...তাৰিখে তেতিয়াৰ অসমৰ মুখ্যমন্ত্ৰী তৰুণ গগৈ..." | Inflected location noun for Assam (অসমৰ) (Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC) |
| #91 | `অসমৰ` | 2 | `O` $\to$ **`B-LOC`** | "...এই সময়ছোৱাত অসমৰ ৰাজনীতিত গোপীনাথ বৰদলৈ..." | Inflected location noun for Assam (অসমৰ) (Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC) |
| #92 | `গুৱাহাটীৰ` | 13 | `O` $\to$ **`B-LOC`** | "...চমুলৈ কলাক্ষেত্ৰ গুৱাহাটীৰ পাঞ্জাবাৰীত অৱস্থিত এক..." | Inflected city location noun (গুৱাহাটীৰ) (Guidelines Section 2.2 & Section 3) |
| #93 | `অসমৰ` | 8 | `O` $\to$ **`B-LOC`** | "...ৰাজনীতিবিদ , অসমৰ প্ৰাক্তন মুখ্যমন্ত্ৰী ।..." | Inflected location noun for Assam (অসমৰ) (Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC) |
| #96 | `অসমৰ` | 3 | `O` $\to$ **`B-LOC`** | "...শইকীয়া , অসমৰ প্ৰাক্তন মুখ্যমন্ত্ৰী..." | Inflected location noun for Assam (অসমৰ) (Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC) |
| #97 | `অসমৰ` | 0 | `O` $\to$ **`B-LOC`** | "...অসমৰ আগশাৰীৰ নাট্যগোষ্ঠী আৱাহন..." | Inflected location noun for Assam (অসমৰ) (Guidelines Section 2.2 & Section 3: অসমলৈ/B-LOC) |
| #99 | `ফণী` | 15 | `O` $\to$ **`B-PER`** | "...ৰাভা আৰু ফণী শৰ্মাই ।..." | Proper personal name of Phani Sarma (Guidelines Section 2.1) |
| #99 | `শৰ্মাই` | 16 | `O` $\to$ **`I-PER`** | "...আৰু ফণী শৰ্মাই ।..." | Surname of Phani Sarma (Guidelines Section 2.1) |
| #104 | `ভাৰতৰ` | 3 | `O` $\to$ **`B-LOC`** | "...মহাকাব্য আৰু ভাৰতৰ সংবিধান দুয়োটাতে উল্লেখ..." | Inflected location noun for India (ভাৰতৰ) (Guidelines Section 2.2 & Section 3) |
| #110 | `বিশ্ববিদ্যালয়ৰ` | 1 | `O` $\to$ **`I-ORG`** | "...গুৱাহাটী বিশ্ববিদ্যালয়ৰ প্ৰথম উপাচাৰ্য..." | University institutional body following named institution stem (Guidelines Section 2.3) |

---

## 5. Step 2c Addendum: Evaluation Benchmark Adjudication (Sentence #200)

> **Context**: Reliability Cycle 2 Step 2c verified Sentence #200 of `long_multientity_eval_v2.conll`.  
> **Source Benchmark**: `data/gold/assamese/long_multientity_eval_v2.conll` (SHA-256: `bce6d69071a26e3180eae58f54e9e22cf66a9bd6389ca5aa51f26fbb5106b17c`).  
> **Adjudicated Versioned Benchmark**: `data/gold/assamese/long_multientity_eval_v2_s200_adjudicated.conll` (SHA-256: `e51942e8ba1cda0d3b86af11e6f2fb0bf64bbadfe63d3f677422179e72414d04`).  
> **Baseline Benchmark Status**: Preserved unmodified; baseline lockfile record unchanged.

### Adjudication Table for Sentence #200 Tokens 8–20:

| Token Index | Token | Original Gold Tag | Adjudicated Gold Tag | Real-World Referent / Entity | Annotation Guideline Section & Justification |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **8** | `ভাৰতীয়` | `B-LOC` | **`B-ORG`** | Reserve Bank of India (ভাৰতীয় ৰিজাৰ্ভ বেংক) | Guidelines Section 1.1 & 2.3: Initial constituent of compound commercial/central bank entity. Must not be fragmented into LOC. |
| **9** | `ৰিজাৰ্ভ` | `O` | **`I-ORG`** | Reserve Bank of India (ভাৰতীয় ৰিজাৰ্ভ বেংক) | Guidelines Section 1.1: Maximal non-overlapping span; inner tokens must not be dropped to non-entity O. |
| **10** | `বেংক` | `B-ORG` | **`I-ORG`** | Reserve Bank of India (ভাৰতীয় ৰিজাৰ্ভ বেংক) | Guidelines Section 1.1: Continuation of compound organization span, not a separate standalone entity. |
| **12** | `ভাৰতীয়` | `B-LOC` | **`B-ORG`** | State Bank of India (ভাৰতীয় ষ্টেট বেংক) | Guidelines Section 1.1 & 2.3: Initial constituent of compound public banking entity. Must not be segmented into LOC. |
| **13** | `ষ্টেট` | `O` | **`I-ORG`** | State Bank of India (ভাৰতীয় ষ্টেট বেংক) | Guidelines Section 1.1: Maximal non-overlapping span; constituent noun must not be dropped to O. |
| **14** | `বেংক` | `B-ORG` | **`I-ORG`** | State Bank of India (ভাৰতীয় ষ্টেট বেংক) | Guidelines Section 1.1: Continuation of compound organization span, not a separate standalone entity. |
| **16** | `ভাৰতীয়` | `B-LOC` | **`B-ORG`** | Life Insurance Corporation of India (ভাৰতীয় জীৱন বিমা নিগম) | Guidelines Section 1.1 & 2.3: Initial constituent of statutory corporation entity. Must not be segmented into LOC. |
| **17** | `জীৱন` | `O` | **`I-ORG`** | Life Insurance Corporation of India (ভাৰতীয় জীৱন বিমা নিগম) | Guidelines Section 1.1: Constituent noun of corporate title must not be dropped to O. |
| **18** | `বিমা` | `O` | **`I-ORG`** | Life Insurance Corporation of India (ভাৰতীয় জীৱন বিমা নিগম) | Guidelines Section 1.1: Constituent noun of corporate title must not be dropped to O. |
| **19** | `নিগম` | `B-ORG` | **`I-ORG`** | Life Insurance Corporation of India (ভাৰতীয় জীৱন বিমা নিগম) | Guidelines Section 1.1: Continuation of compound organization span, not a separate standalone entity. |

