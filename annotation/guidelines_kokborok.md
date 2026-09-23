# Named Entity Annotation Guidelines: Kokborok (Roman Script)

**Language**: Kokborok (Tripuri / `trp`)  
**Script**: Roman (Latin) Orthography  
**Standard**: CoNLL-2003 BIO (PER, LOC, ORG, MISC)  
**Authority**: Ground-truth annotation manual for human annotators and adjudicators.

---

## 1. General Principles

1. **Maximal Non-Overlapping Spans**: Annotate the full contiguous named entity span without nesting (e.g., `[ORG Tata Technologies]`, not nested).
2. **Punctuation Exclusion**: Commas, periods, exclamation marks, question marks, and quotation marks are strictly tagged as `O`.
3. **Agglutinative Suffix Integration**: In Kokborok, case markers (`-o`, `-no`, `-ni`, `-bai`) and plural markers (`-rok`) that are bound to an entity token (either directly or via hyphen) remain part of that entity token.
   - Example: `Tripurao` $\to$ `[B-LOC Tripurao]`
   - Example: `Agartala Station-o` $\to$ `[B-LOC Agartala] [I-LOC Station-o]`

---

## 2. Entity Classes & Disambiguation Rules

### 2.1 Person (`PER`)
- **Proper Names**: Personal names, surnames, and patronymics:
  - `Mohan/B-PER`
  - `kanika/B-PER`
  - `Bir/B-PER Bikram/I-PER`
  - `Radhakishore/B-PER Manikya/I-PER`
  - `Alindra/B-PER Debbarma/I-PER`
  - `Naphurai/B-PER Jamatia/I-PER`
- **Honorifics and Titles**: Common titles and kinship markers (`Boromgwnang`, `Sir`, `Madam`, `Dr.`, `Shri`, `Raja`) are tagged as `O`:
  - `Boromgwnang/O Mohan/B-PER`
  - `Dr./O Debbarma/B-PER`
- **Pronouns and Common Nouns**: Kokborok pronouns (`Ang`, `Nung`, `Bo`, `Chwng`, `Nini`, `Ani`) are strictly `O`.

### 2.2 Location (`LOC`)
- **Geopolitical & Topographical Entities**: States, districts, cities, towns, villages, rivers, hills, and stations:
  - `Tripura/B-LOC`
  - `Agartala/B-LOC`
  - `Kailashahar-o/B-LOC`
  - `Bilonia-o/B-LOC`
  - `Udaipur/B-LOC`
  - `Dharmanagar/B-LOC`
  - `Khowai/B-LOC`
  - `Ambassa/B-LOC`
  - `Jampui/B-LOC Hills/I-LOC`
  - `Howrah/B-LOC Twima/I-LOC` (Howrah river)
  - `Gomati/B-LOC Twima/I-LOC`
  - `Agartala/B-LOC Station-o/I-LOC`
- **Inflected Locatives**:
  - `Tripurao/B-LOC` (in Tripura)
  - `Agartala-ni/B-LOC` (of Agartala)
  - `Kailashahar-o/B-LOC` (in Kailashahar)

### 2.3 Organization (`ORG`)
- **Commercial, Governmental, Academic, and Cultural Bodies**:
  - `Tata/B-ORG Technologies/I-ORG`
  - `Tripura/B-ORG University/I-ORG`
  - `Tripura/B-ORG Legislative/I-ORG Assembly/I-ORG`
  - `Twipra/B-ORG Students/I-ORG Federation/I-ORG` (TSF)
  - `Kokborok/B-ORG Tei/I-ORG Hukumu/I-ORG Mission/I-ORG` (KTHM)
  - `Tripura/B-ORG Board/I-ORG`
  - `State/B-ORG Bank/I-ORG of/I-ORG India/I-ORG`
- **Government as Actor**:
  - `Tripura/B-ORG signs MoA with Tata Technologies.` $\to$ `Tripura/B-ORG` (acting as state government signatory), `Tata/B-ORG Technologies/I-ORG`.

### 2.4 Miscellaneous (`MISC`)
- **Festivals, Cultural Ceremonies, Treaties, Laws, and Demonyms**:
  - `Kharchi/B-MISC Puja/I-MISC`
  - `Garia/B-MISC Puja/I-MISC`
  - `Ker/B-MISC Puja/I-MISC`
  - `Kokborok/B-MISC Sal/I-MISC` (Kokborok Language Day)
  - `MoA/B-MISC` (Memorandum of Agreement)
  - `Bangladeshirok/B-MISC` (Demonym / Nationality with plural suffix `-rok`)
  - `Kokborok/B-MISC` (When referring to the language entity itself)

---

## 3. Disambiguation Matrix for Kokborok

| Example Context | Annotated Span | Disambiguation Rationale |
| :--- | :--- | :--- |
| `Tata Technoligies bai Tripura kaisa MoA gosimung rwlaikha.` | `Tata/B-ORG Technoligies/I-ORG`, `Tripura/B-ORG`, `MoA/B-MISC` | `Tripura` here is the institutional signatory (Government); `MoA` is formal agreement. |
| `Belai watwi wama bagwi Tripurao twilom kha.` | `Tripurao/B-LOC` | Spatial/geographical state territory subject to flooding. Locative `-o` bound to stem. |
| `Agartala Station-o Khoroksara Bangladeshirok romjak kha.` | `Agartala/B-LOC Station-o/I-LOC`, `Bangladeshirok/B-MISC` | Multi-word infrastructure toponym; `Bangladeshirok` is plural demonym. |
| `Where is Mohan Bo phaiwi de tong` | `Mohan/B-PER` | Personal name in conversational query. |
| `Tinile ang chicken bai pork songwi chasinai` | All tokens `O` | Food/meat items are common nouns, not named entities. |
| `Ang bank-o thanani nangnai.` | `bank-o/O` | General facility, not a named institution (unlike `State Bank of India`). |
