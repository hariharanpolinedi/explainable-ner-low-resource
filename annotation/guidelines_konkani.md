# Named Entity Annotation Guidelines: Konkani (Devanagari)

**Language**: Konkani (`gom`)  
**Script**: Devanagari  
**Standard**: CoNLL BIO (PER, LOC, ORG, MISC)  

---

## 1. General Principles

1. **Maximal Non-Overlapping Spans**: Annotate the full contiguous named entity span without nesting.
2. **Punctuation Exclusion**: Trailing dandas (`।`), commas (`,`), periods (`.`), and quotation marks are strictly tagged as `O`.

---

## 2. Entity Classes & Disambiguation Rules

### 2.1 Person (`PER`)
- **Proper Names**: Single or multi-word personal names:
  - `दामोदर/B-PER मावजो/I-PER`
  - `मनोहर/B-PER पर्रीकार/I-PER`
- **Honorifics and Titles**: Common titles (`डॉ.`, `श्री.`, `श्रीमती.`, `प्रा.`) are tagged as `O`:
  - `डॉ./O आंबेडकरांनी/B-PER`
  - `श्री./O मनोहर/B-PER राय/I-PER सरदेसाय/I-PER`
- **Lexicalized Epithets**: Fixed historical pseudonyms/epithets where the title is inseparable:
  - `शणै/B-PER गोंयबाब/I-PER`
  - `बाकीबाब/B-PER बोरकार/I-PER`

### 2.2 Location (`LOC`)
- **Geopolitical & Natural Entities**: Countries, states, districts, talukas, cities, villages, rivers, beaches, hills:
  - `गोंय/B-LOC`
  - `उत्तर/B-LOC गोंय/I-LOC`
  - `पणजी/B-LOC`
  - `मडगांव/B-LOC`
  - `मांडवी/B-LOC न्हंय/I-LOC`
  - `कळंगूट/B-LOC दर्यावेळ/I-LOC`

### 2.3 Organization (`ORG`)
- **Formal Institutions & Bodies**: Government departments, universities, political parties, media companies, academies:
  - `गोंय/B-ORG विद्यापीठ/I-ORG`
  - `कोंकणी/B-ORG भाशा/I-ORG मंडळ/I-ORG`
  - `गोवा/B-ORG विधानसभा/I-ORG`
  - `भांगरभूंय/B-ORG` (newspaper organization)
- **Nested Person in ORG**: If a proper person name forms part of an organization name, tag the whole span as ORG:
  - `रवींद्र/B-ORG भवन/I-ORG` (not `रवींद्र/B-PER`)

### 2.4 Miscellaneous (`MISC`)
- **Festivals, Events, Historical Acts & Awards**:
  - `शिगमो/B-MISC`
  - `चवथ/B-MISC`
  - `साहित्य/B-MISC अकादेमी/I-MISC पुरस्कार/I-MISC`
  - `गोंय/B-MISC मुक्ती/I-MISC दीस/I-MISC`

---

## 3. Ambiguity Resolution Matrix

| Example Text | Annotated Span | Rationale |
| :--- | :--- | :--- |
| `गोंय सरकारान आदेश जारी केलो` | `गोंय/B-ORG सरकारान/I-ORG` | Government as an institutional governing body |
| `गोंयांत जायते पर्यटक येतात` | `गोंयांत/B-LOC` | Goa as geographical territory |
| `डॉ. प्रमोद सावंत हांणी उलयलें` | `डॉ./O प्रमोद/B-PER सावंत/I-PER` | Title excluded from person span |
| `कला अकादमी पणजी शारांत आसा` | `कला/B-ORG अकादमी/I-ORG पणजी/B-LOC` | Organization followed by separate location |
