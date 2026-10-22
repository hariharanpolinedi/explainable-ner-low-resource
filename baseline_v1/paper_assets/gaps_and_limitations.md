# Scientific Gaps, Unresolved Failure Modes & Methodological Limitations

**Paper Section Target**: Discussion / Limitations / Threats to Validity  
**Scope**: Explicit documentation of architectural boundaries, unresolved linguistic edge cases, and comparison disparities.

---

## 1. The ChatGPT / Commercial LLM Comparison Gap

A common reviewer question in contemporary NLP is: *"How does this system compare against commercial Large Language Models such as ChatGPT (GPT-4o) or Claude 3.5 Sonnet?"*

In this research project, an empirical benchmark against closed commercial LLMs was intentionally omitted. This omission is **structurally expected and scientifically motivated, rather than an experimental oversight**:

1. **Fundamental Constraint Asymmetry**:
   - Commercial LLMs are pretrained on trillions of tokens of multilingual web text with hundreds of billions of parameters, utilizing massive compute clusters.
   - This project investigates an orthogonal scientific question: *Can a compact, from-scratch neural architecture (10.6M parameters) learn competitive, structured entity representations for low-resource languages using strictly local corpora (~50,000 sentences) with zero pretrained models?*
   - Direct comparison against models trained on massive, uncontrolled pretraining corpora invalidates the low-resource from-scratch hypothesis.
2. **Inherent Incompatibility with Attention Attribution**:
   - The central explainability contribution of this work is **Gradient $\times$ Attention attribution** ($\nabla_{\alpha} \text{score}$), which requires full, white-box tensor access to multi-head attention matrices and exact backward autograd passes.
   - Commercial LLM APIs (OpenAI, Anthropic, Google Gemini) operate as black boxes, exposing only generated token strings or top-$k$ logprobs. They do not expose intermediate self-attention activation maps or support backward gradient computation with respect to internal layers.
3. **Severe Tokenizer Fragmentation in Low-Resource Indic Scripts**:
   - Web-scale LLM tokenizers (e.g., `tiktoken` / `cl100k_base`) are predominantly trained on English/Latin text. In Konkani and Sinhala, standard subword tokenizers decompose single Brahmic aksharas into 3 to 6 raw UTF-8 byte tokens. This induces extreme sequence dilation, high latency, context degradation, and catastrophic boundary hallucinations.
4. **Data Sovereignty & Offline Deployment**:
   - In public administration and regional language settings, reliance on proprietary, US-hosted cloud APIs presents severe privacy, regulatory, and cost bottlenecks. A 10.6M parameter model runs entirely offline on standard consumer hardware (e.g., laptop RTX 3050) with sub-10ms inference.

---

## 2. Unresolved Inflectional-OOV & Orthographic Failure Modes

Despite achieving >96% test F1 across all three languages, specific linguistic and orthographic failure modes remain unresolved:

### 2.1 Konkani Case Agglutination & Postposition Binding
- **Failure Mode**: Konkani places take bound locative (`-ंत`, e.g., `पणजी` $\to$ `पणजींत` [in Panaji]) and genitive inflections (`-चो/-च्या`, e.g., `गोवा` $\to$ `गोयाच्या` [of Goa]).
- **Mechanism**: The gazetteer and weak supervision rules only match nominative base lemmas (`पणजी`, `गोवा`). When inflected surface variants appear in running text, they are frequently OOV in the word embedding vocabulary ($100\text{d}$).
- **Observed Impact**: While the Character-CNN extracts partial morphological cues, the linear-chain CRF occasionally drops the token or misclassifies the span as background `O`. On Gold-Test, Konkani PER recall lagged at **$85.00\%$** (3 false negatives out of 20 gold instances), predominantly on inflected personal surnames and compound titles.

### 2.2 Sinhala Zero-Width Joiner (ZWJ) Conjunct Orthography
- **Failure Mode**: In Sinhala script, consonant conjuncts require an explicit Unicode Zero-Width Joiner (`\u200D`, ZWJ) between the virama (hal kirima `\u0DCA`) and the conjunct consonant (e.g., `ශ්` + `\u200D` + `රී` = `ශ්‍රී`).
- **Mechanism**: When raw web text or user input decomposes or strips the ZWJ (`ශ්රී`), the grapheme cluster tokenizer decomposes the akshara into separate orthographic clusters (`ප්`, `ර`).
- **Empirical Impact**: As verified in [`reports/zwj_hypothesis_verification.md`](file:///c:/Users/harih/Desktop/NLP-MARK-3/reports/zwj_hypothesis_verification.md), omitting ZWJ causes a **100% entity detection collapse** on base named entities (e.g., `ශ්‍රී ලංකාව` is correctly detected as LOC, but `ශ්රී ලංකාව` without ZWJ collapses completely to `O`).
- **Resolution Status**: Retraining on perturbed ZWJ data regressed overall Dev F1 from $99.24\%$ to $98.13\%$. The issue is resolved via strict input Unicode NFC/ZWJ canonicalization in the preprocessing pipeline, but remains an inherent vulnerability if unnormalized text reaches the model.

### 2.3 Sinhala Locative & Genitive Suffix Agglutination
- **Failure Mode**: Locative markers (`-යේදී`, e.g., `විශ්වවිද්‍යාලයේදී` [at the university]) and genitive markers (`-වේ`, e.g., `ශ්‍රී ලංකාවේ` [of Sri Lanka]).
- **Observed Impact**: In long sentences with dense conjunctions, inflected multi-word organizations suffer boundary truncation, where the base noun is detected but trailing inflected tokens are misclassified.

---

## 3. Honest Accounting of Sub-Slice Performance Below >90% Target

On the official locked gold-test evaluation, all three languages surpassed the >90.0% macro research target:
- **Konkani**: 96.27% Entity F1 (+6.27 pp margin)
- **Assamese**: 97.87% Entity F1 (+7.87 pp margin)
- **Sinhala**: 97.76% Entity F1 (+7.76 pp margin)

However, an honest accounting reveals several specific sub-slices and intermediate configurations that fell below the >90.0% threshold:

| Evaluation Slice / Condition | Language | Metric / Entity Type | Honest Observed Number | Deficit vs >90% Target | Root Cause |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Gold-Test Sub-Slice** | Konkani | **PER Recall** | **85.00%** | **-5.00 pp** | Tail surname inflections; 3 missed entities out of 20. |
| **Gold-Test Sub-Slice** | Assamese | **PER Precision** | **86.36%** | **-3.64 pp** | 3 false positives where model detected valid entities unannotated in gold test. |
| **Gold-Test Sub-Slice** | Assamese | **Out-of-Vocabulary (OOV) F1** | **86.67%** | **-3.33 pp** | Lexical memorization gap on entities completely absent from gold-train. |
| **Out-of-Sample Weak Supervision** | Assamese | **ORG Silver Precision** | **81.25%** | **-8.75 pp** | Rule noise on multi-word organization titles evaluated out-of-sample on Gold-Train. |
| **Out-of-Sample Weak Supervision** | Konkani | **ORG Silver Precision** | **86.90%** | **-3.10 pp** | Heuristic mismatch on compound institutions. |
| **Ablation: No Self-Attention** | Sinhala | **Overall Dev F1** | **51.45%** | **-38.55 pp** | Catastrophic collapse of boundary synchronization without attention. |
| **Ablation: No Self-Attention** | Konkani | **Overall Dev F1** | **58.58%** | **-31.42 pp** | Severe degradation in long-range sequence context. |
| **Ablation: No Self-Attention** | Assamese | **Overall Dev F1** | **72.59%** | **-17.41 pp** | Loss of multi-head dependency tracking. |
| **Ablation: No Char-CNN** | Sinhala | **Overall Dev F1** | **83.20%** | **-6.80 pp** | Inability to generalize to unseen morphologically inflected words. |
| **Ablation: No Char-CNN** | Konkani | **Overall Dev F1** | **87.66%** | **-2.34 pp** | Subword boundary loss on Devanagari ligatures. |
| **Ablation: No Char-CNN** | Assamese | **Overall Dev F1** | **88.89%** | **-1.11 pp** | Inflectional sensitivity on Eastern Nagari case endings. |
| **Ablation: No CRF** | Sinhala | **Overall Dev F1** | **76.11%** | **-13.89 pp** | Rampant illegal BIO transitions without sequence transition matrix. |
| **Ablation: Subword BPE Tokenizer** | Sinhala | **Overall Dev F1** | **81.70%** | **-8.30 pp** | BPE fragmentation of Sinhala consonant-vowel akshara ligatures. |

---

## 4. Corpus-Size Asymmetry Across Target Languages

A primary data-centric limitation is the marked asymmetry in raw corpus availability across the three target languages:

```
Sinhala:  66,954 sentences ──┐
                             ├── Asymmetry Factor: 1.30x (Sinhala vs. Konkani)
Assamese: 55,002 sentences ──┤
                             └── Wikipedia Shortfall: Konkani required newspaper scrape
Konkani:  51,488 sentences ──┘
```

1. **Corpus Sourcing Disparity**:
   - **Sinhala (66,954 sents)** and **Assamese (55,002 sents)** readily met the 50,000-sentence clean data floor from native Wikipedia dumps alone (66,887 and 54,893 sentences respectively).
   - **Konkani (51,488 sents)** suffered a severe data deficit on Wikipedia, yielding only 30,798 clean sentences. To reach the required 50,000-sentence floor, an emergency secondary harvest of 20,670 sentences was conducted from regional Devanagari press archives (*Bhaangarbhuin* newspaper).
2. **Impact on Error Distribution**:
   - In Phase 14 error analysis, **88.89% of all Konkani test errors were False Negatives (missed entities)**.
   - Konkani's smaller lexical base (58,162 unique words vs 76,826 in Sinhala) led to higher sensitivity to out-of-vocabulary entities (In-Vocab F1: 97.09% vs OOV F1: 91.43%).
   - While multilingual joint warm-start in Phase 8 provided significant cross-lingual regularization (+2.52% net gain for Konkani), multilingual transfer cannot entirely substitute for native lexical frequency in low-resource orthographies.
