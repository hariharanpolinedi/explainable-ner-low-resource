# Phase 7: Fresh-Sentence Generalization Check

> **Phase 7 Deliverable — Out-of-Distribution Generalization Audit**  
> Evaluated on 6 genuinely fresh sentences (2 per language) with 100% verified non-overlap with all training/eval splits.

---

## 1. Non-Overlap Verification Audit

Every test sentence was checked against all gold, silver, pseudo, augmented, and stress-set ConLL files.
- **Konkani**: 0 matching sentences (Verified 100% Fresh).
- **Assamese**: 0 matching sentences (Verified 100% Fresh).
- **Sinhala**: 0 matching sentences (Verified 100% Fresh).

---

## 2. Generalization Performance by Language

### Konkani

**Sentence 1**: `काल्ह मडगांवां जाल्ल्या एका सुवाळ्यांत डॉ. सुदेश कामत हांणी गोवा विद्यापीठाच्या विद्यार्थ्यांक संबोधित केलें .`
- **Gold Entities**: [('LOC', 1, 2), ('PER', 6, 8), ('ORG', 9, 11)]
- **Predicted Entities**: []
- **Accuracy**: 0 / 3 entities detected

**Sentence 2**: `वास्को दा गामा शारांत नवी दिल्ली सावन आयिल्ल्या भारतीय जनता पार्टीच्या फुड्यांनी व्हड रॅली काडली .`
- **Gold Entities**: [('LOC', 0, 3), ('LOC', 4, 6), ('ORG', 8, 11)]
- **Predicted Entities**: [('LOC', 0, 1)]
- **Accuracy**: 0 / 3 entities detected

### Assamese

**Sentence 1**: `যোৱাকালি গুৱাহাটীত অনুস্থিত এখন সভাত মুখ্যমন্ত্ৰী হিমন্ত বিশ্ব শৰ্মা আৰু কেন্দ্ৰীয় মন্ত্ৰী সৰ্বানন্দ সোণোৱাল উপস্থিত আছিল .`
- **Gold Entities**: [('LOC', 1, 2), ('PER', 6, 9), ('PER', 12, 14)]
- **Predicted Entities**: [('PER', 6, 9), ('PER', 12, 14)]
- **Accuracy**: 2 / 3 entities detected

**Sentence 2**: `বৰপেটা চহৰত অসম আৰক্ষীৰ এটা দলে এক বিশেষ অভিযান চলাই তিনিজন সন্দেহযুক্ত লোকক আটক কৰে .`
- **Gold Entities**: [('LOC', 0, 1), ('ORG', 2, 4)]
- **Predicted Entities**: [('LOC', 0, 1), ('LOC', 2, 3)]
- **Accuracy**: 1 / 2 entities detected

### Sinhala

**Sentence 1**: `ඊයේ මහනුවර නගරයේ පැවති උත්සවයකදී ජනාධිපති අනුර කුමාර දිසානායක මහතා විශේෂ ප්‍රකාශයක් කළේය .`
- **Gold Entities**: [('LOC', 1, 2), ('PER', 6, 9)]
- **Predicted Entities**: [('PER', 6, 9), ('LOC', 1, 2)]
- **Accuracy**: 2 / 2 entities detected

**Sentence 2**: `කොළඹ විශ්වවිද්‍යාලය සහ ශ්‍රී ලංකා මහ බැංකුව එක්ව නව ආර්ථික සමුළුවක් ගාලු මුවදොර හෝටලයේදී පැවැත්වීය .`
- **Gold Entities**: [('ORG', 0, 2), ('ORG', 3, 7), ('LOC', 11, 14)]
- **Predicted Entities**: [('ORG', 3, 7), ('ORG', 0, 2)]
- **Accuracy**: 2 / 3 entities detected

