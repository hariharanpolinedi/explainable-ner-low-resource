# Named Entity Annotation Guidelines: Assamese

**Language**: Assamese (`as`)  
**Script**: Bengali-Assamese (Eastern Nagari)  
**Standard**: CoNLL BIO (PER, LOC, ORG, MISC)  

---

## 1. General Principles

1. **Maximal Non-Overlapping Spans**: Annotate complete compound entity mentions without inner segmentation.
2. **Punctuation Exclusion**: Trailing dandas (`।`), commas (`,`), and colons are tagged `O`.

---

## 2. Entity Classes & Disambiguation Rules

### 2.1 Person (`PER`)
- **Proper Names**:
  - `লক্ষ্মীনাথ/B-PER বেজবৰুৱা/I-PER`
  - `শ্ৰীমন্ত/B-PER শংকৰদেৱ/I-PER`
  - `লাচিত/B-PER বৰফুকন/I-PER`
  - `ভূপেন/B-PER হাজৰিকা/I-PER`
- **Titles & Honorifics**: Standalone titles (`ডাঃ`, `ড°`, `প্ৰাঃ`, `শ্ৰী`, `শ্ৰীমতী`) are tagged `O`:
  - `ড°/O ভূপেন/B-PER হাজৰিকা/I-PER`
  - `শ্ৰী/O হিমন্ত/B-PER বিশ্ব/I-PER শৰ্মা/I-PER`

### 2.2 Location (`LOC`)
- **Geographical, Administrative, Topographical Entities**:
  - `অসম/B-LOC`
  - `গুৱাহাটী/B-LOC`
  - `ব্ৰহ্মপুত্ৰ/B-LOC নদী/I-LOC`
  - `কাজিৰঙা/B-LOC ৰাষ্ট্ৰীয়/I-LOC উদ্যান/I-LOC`
  - `মাজুলী/B-LOC`

### 2.3 Organization (`ORG`)
- **Societies, Institutions, Universities, Commercial Enterprises, Political Parties**:
  - `অসম/B-ORG সাহিত্য/I-ORG সভা/I-ORG`
  - `গুৱাহাটী/B-ORG বিশ্ববিদ্যালয়/I-ORG`
  - `অসম/B-ORG বিধানসভা/I-ORG`
  - `অইল/B-ORG ইণ্ডিয়া/I-ORG লিমিটেড/I-ORG`
  - `নিয়মীয়া/B-ORG বাৰ্তা/I-ORG` (newspaper publisher)
- **Nested Names**: When an institution bears a personal name, tag as single ORG:
  - `শ্ৰীমন্ত/B-ORG শংকৰদেৱ/I-ORG কলাক্ষেত্ৰ/I-ORG` (not `PER`)

### 2.4 Miscellaneous (`MISC`)
- **Festivals, Historical Events, Treaties, Awards**:
  - `ৰঙালী/B-MISC বিহু/I-MISC`
  - `ভোগালী/B-MISC বিহু/I-MISC`
  - `বুঢ়ী/B-MISC আইৰ/I-MISC সাধু/I-MISC` (book title)
  - `অসম/B-MISC চুক্তি/I-MISC` (Assam Accord)
  - `পদ্মশ্ৰী/B-MISC`

---

## 3. Ambiguity Resolution Matrix

| Example Text | Annotated Span | Rationale |
| :--- | :--- | :--- |
| `অসম চৰকাৰৰ উদ্যোগত` | `অসম/B-ORG চৰকাৰৰ/I-ORG` | Government as political administration (ORG) |
| `অসমলৈ আহিছিল` | `অসমলৈ/B-LOC` | Inflected location noun |
| `কাজিৰঙা উদ্যানত বাঘ আছে` | `কাজিৰঙা/B-LOC উদ্যানত/I-LOC` | National park entity |
| `অসম সাহিত্য সভাত যোগ দিলে` | `অসম/B-ORG সাহিত্য/I-ORG সভাত/I-ORG` | Institution governing body |
