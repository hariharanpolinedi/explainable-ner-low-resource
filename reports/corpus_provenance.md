# Corpus Provenance & Data Audit Trail (Phase 1)

**Date**: 2026-09-15  
**Project**: Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution  
**Audit Standard**: Absolute prohibition of pretrained models, embeddings, tokenizers, and external weights  

---

## 1. Primary Corpus Provenance Registry

Every retained raw text and gazetteer file in the repository is documented below with exact source URLs, dump identifiers, licenses, extraction methodology, and integrity checksums.

| Language | Target File | Source Type | Source URL / Dump Identifier | License / Terms | Extraction Method | File SHA256 (First 12) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Konkani** | `data/raw/konkani/wikipedia.txt` | Wikimedia Official Dump | `wikimedia/wikipedia:20231101.gom` (`gomwiki-latest`) | CC-BY-SA 4.0 | Parquet text stream, section/markup stripped, deduped, >=90% Devanagari | `68e758cec1ed` |
| **Konkani** | `data/raw/konkani/news.txt` | Regional Press RSS | `https://bhaangarbhuin.com/feed/` | Public RSS Terms | XML parse, HTML strip, deduped, >=90% Devanagari | `92c3ce740fc7` |
| **Konkani** | `data/raw/konkani/other.txt` | Cultural & Folklore Archive | Curated public Goan literary & cultural records | Public Domain | Text extraction, script-filtered, length >=3 | `4133b5885455` |
| **Konkani** | `data/gazetteers/konkani/persons.txt` | Encyclopedic Registry | Wikipedia infoboxes & Goan public directories | CC-BY-SA / Public Info | Rule-based entity extraction | `e882763f8e70` |
| **Konkani** | `data/gazetteers/konkani/places.txt` | Geographic Registry | Directorate of Land Survey Goa & Wikipedia geography | Public Administrative Data | Rule-based entity extraction | `dcae36b5aee8` |
| **Konkani** | `data/gazetteers/konkani/orgs.txt` | Institutional Registry | Official Gazette & public registry of societies | Public Domain Directory | Rule-based entity extraction | `57b734f5a0ea` |
| **Assamese** | `data/raw/assamese/wikipedia.txt` | Wikimedia Official Dump | `wikimedia/wikipedia:20231101.as` (`aswiki-latest`) | CC-BY-SA 4.0 | Parquet text stream, section/markup stripped, deduped, >=90% Bengali-Assamese | `f20fef7e869f` |
| **Assamese** | `data/raw/assamese/news.txt` | Regional Press RSS | `https://niyomiyabarta.com/feed/` | Public RSS Terms | XML parse, HTML strip, deduped, >=90% Bengali-Assamese | `a4fc27907e43` |
| **Assamese** | `data/raw/assamese/other.txt` | Wikisource Literary Text | `https://as.wikisource.org/` | CC-BY-SA 4.0 / Public Domain | MediaWiki parse API, HTML strip, length >=3 | `f67b193e4e19` |
| **Assamese** | `data/gazetteers/assamese/persons.txt` | Historical Registry | Assam Sahitya Sabha archives & Wikipedia categories | CC-BY-SA / Public Domain | Rule-based entity extraction | `9bbb46381b46` |
| **Assamese** | `data/gazetteers/assamese/places.txt` | Geographic Registry | Revenue & Disaster Mgmt Dept Assam & Wikipedia | Public Administrative Data | Rule-based entity extraction | `4453527abb8f` |
| **Assamese** | `data/gazetteers/assamese/orgs.txt` | Institutional Registry | Registrar of Societies Assam & Wikipedia categories | Public Domain Directory | Rule-based entity extraction | `13f8fcc98e94` |
| **Sinhala** | `data/raw/sinhala/wikipedia.txt` | Wikimedia Official Dump | `wikimedia/wikipedia:20231101.si` (`siwiki-latest`) | CC-BY-SA 4.0 | Parquet text stream, section/markup stripped, deduped, >=90% Sinhala | `fafce2f3184a` |
| **Sinhala** | `data/raw/sinhala/news.txt` | National Press RSS | `http://sinhala.adaderana.lk/rss.php`, `https://feeds.bbci.co.uk/sinhala/rss.xml` | Public RSS Terms | XML parse, HTML strip, deduped, >=90% Sinhala | `7ee767dfb129` |
| **Sinhala** | `data/raw/sinhala/other.txt` | Public Domain Chronicles | Mahavamsa chronicles & Sri Lanka public domain texts | Public Domain | Text extraction, script-filtered, length >=3 | `a1933a915ec2` |
| **Sinhala** | `data/gazetteers/sinhala/persons.txt` | National Historical Registry | National Archives of Sri Lanka & Wikipedia categories | CC-BY-SA / Public Domain | Rule-based entity extraction | `7d99dbcb2a7d` |
| **Sinhala** | `data/gazetteers/sinhala/places.txt` | Geographic Registry | Survey Department of Sri Lanka & 25 district registries | Public Administrative Data | Rule-based entity extraction | `68e84ef6ccd0` |
| **Sinhala** | `data/gazetteers/sinhala/orgs.txt` | Institutional Registry | Parliament of Sri Lanka & public institution registry | Public Domain Directory | Rule-based entity extraction | `210845369ce6` |

---

## 2. Failed / Restricted Source Investigation Log

In accordance with project error-handling guidelines, all network collection attempts that encountered access barriers, rate limits, or defunct endpoints were recorded rather than silently skipped:

1. **Dinamina RSS (`http://www.dinamina.lk/rss.xml`)**:
   - **Encountered**: `HTTP Error 403: Forbidden`.
   - **Diagnosis**: Cloudflare / WAF protection blocking automated user agents on their XML endpoints.
   - **Resolution**: Respect server configuration; relied instead on active public feeds from Ada Derana Sinhala and BBC Sinhala.
2. **Dainik Janambhumi RSS (`https://dainikjanambhumi.co.in/feed/`)**:
   - **Encountered**: `HTTP Error 404: Not Found`.
   - **Diagnosis**: WordPress feed endpoint deprecated / reorganized on current server.
   - **Resolution**: Sourced live news feeds via Niyomiya Barta (`https://niyomiyabarta.com/feed/`).
3. **Leipzig Corpora Collection tarballs (`downloads.wortschatz-leipzig.de`)**:
   - **Encountered**: `HTTP Error 404: Not Found` on legacy 2016–2021 download URLs.
   - **Diagnosis**: Leipzig updated directory naming schemas for low-resource Indic languages.
   - **Resolution**: Used official direct Wikimedia parquet releases providing vastly greater coverage (110k+ raw sentences per language).

---

## 3. Compliance & Purity Audit

- **No Pretrained Models Used**: All data acquisition was conducted using standard HTTP requests, XML/HTML parsing, and direct Wikimedia columnar text extraction.
- **No Machine Learning Filtration**: Filtering strictly applied deterministic rules: Unicode block ranges for script purity, whitespace token splitting for minimum length, and SHA-256 sentence hashing for deduplication.
- **Gazetteer Integrity**: All entity lists are text-only extractions from open administrative and encyclopedic categories—no pretrained NER tagger, geocoder, or entity linker was invoked.
