# Named Entity Annotation Guidelines: Sinhala

**Language**: Sinhala (`si`)  
**Script**: Sinhala  
**Standard**: CoNLL BIO (PER, LOC, ORG, MISC)  

---

## 1. General Principles

1. **Maximal Non-Overlapping Spans**: Annotate the full contiguous named entity span without nesting.
2. **Punctuation Exclusion**: Trailing periods (`.`), kunddaliya (`෴`), and commas are tagged as `O`.
3. **Zero-Width Joiners (ZWJ)**: Internal ZWJ conjuncts (`ශ්‍රී`, `ක්‍ෂ`) are preserved inside word tokens.

---

## 2. Entity Classes & Disambiguation Rules

### 2.1 Person (`PER`)
- **Proper Names**:
  - `මාර්ටින්/B-PER වික්‍රමසිංහ/I-PER`
  - `ඩබ්ලිව්./B-PER ඩී./I-PER අමරදේව/I-PER`
  - `දුටුගැමුණු/B-PER රජු/I-PER` (monarch epithet)
  - `මහින්ද/B-PER රාජපක්ෂ/I-PER`
- **Honorifics and Titles**: Standard titles (`මහාචාර්ය`, `ඩොක්ටර්`, `ගරු`, `පූජ්‍ය`) are tagged as `O`:
  - `මහාචාර්ය/O සෙනරත්/B-PER පරණවිතාන/I-PER`
  - `ගරු/O අගමැති/O රනිල්/B-PER වික්‍රමසිංහ/I-PER`

### 2.2 Location (`LOC`)
- **Geopolitical & Natural Entities**: Countries, provinces, districts, cities, rivers, mountains:
  - `ශ්‍රී/B-LOC ලංකාව/I-LOC`
  - `කොළඹ/B-LOC`
  - `මහනුවර/B-LOC`
  - `මහවැලි/B-LOC ගඟ/I-LOC`
  - `සීගිරිය/B-LOC`

### 2.3 Organization (`ORG`)
- **Institutions, Universities, Government Departments, Corporations, Media**:
  - `ශ්‍රී/B-ORG ලංකා/I-ORG පාර්ලිමේන්තුව/I-ORG`
  - `කොළඹ/B-ORG විශ්වවිද්‍යාලය/I-ORG`
  - `ශ්‍රී/B-ORG ලංකා/I-ORG මහ/I-ORG බැංකුව/I-ORG`
  - `දිනමිණ/B-ORG` (newspaper publication)
  - `අද/B-ORG දෙරණ/I-ORG` (media agency)
- **Nested Person in ORG**:
  - `බණ්ඩාරනායක/B-ORG සම්මන්ත්‍රණ/I-ORG ශාලාව/I-ORG` (BMICH tagged as ORG)

### 2.4 Miscellaneous (`MISC`)
- **Historical Chronicles, Treaties, Cultural Festivals, Awards**:
  - `මහාවංශය/B-MISC`
  - `සිංහල/B-MISC අලුත්/I-MISC අවුරුද්ද/I-MISC`
  - `දළදා/B-MISC පෙරහැර/I-MISC`
  - `රණවිරු/B-MISC සම්මානය/I-MISC`

---

## 3. Ambiguity Resolution Matrix

| Example Text | Annotated Span | Rationale |
| :--- | :--- | :--- |
| `ශ්‍රී ලංකා රජය විසින්` | `ශ්‍රී/B-ORG ලංකා/I-ORG රජය/I-ORG` | Government as administrative organ (ORG) |
| `ශ්‍රී ලංකාවට පැමිණියේය` | `ශ්‍රී/B-LOC ලංකාවට/I-LOC` | Country as geographical destination |
| `මහාචාර්ය එදිරිවීර සරච්චන්ද්‍ර` | `මහාචාර්ය/O එදිරිවීර/B-PER සරච්චන්ද්‍ර/I-PER` | Academic title excluded from name |
| `පේරාදෙණිය විශ්වවිද්‍යාලය මහනුවර පිහිටා ඇත` | `පේරාදෙණිය/B-ORG විශ්වවිද්‍යාලය/I-ORG මහනුවර/B-LOC` | Institution followed by city |
