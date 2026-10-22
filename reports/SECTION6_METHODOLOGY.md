# SECTION 6 — METHODOLOGY (CORRECTED)

---

## 6.1 Data Collection and Dataset Preparation

This study addresses the problem of named entity recognition (NER) across four low-resource
languages: Konkani, Assamese, Sinhala, and Kokborok. Each language presents distinct
typological and orthographic characteristics. Konkani and Assamese are Indo-Aryan languages
written in the Devanagari and Assamese-Bengali scripts, respectively. Sinhala is an
Indo-Aryan language of Sri Lanka written in its own abugida script. Kokborok is a
Tibeto-Burman agglutinative language from the North-East Indian state of Tripura, written
in the Roman script for the purposes of this study. The combination of four structurally
diverse languages, each employing a different script or orthographic convention, establishes
a challenging and representative multilingual setting for low-resource NER research.

Raw text data for Konkani, Assamese, and Sinhala was collected from publicly available
Wikipedia dumps and online news sources including Herald Goa, Asomiya Pratidin, and Ada
Derana Sinhala, filtered to remove documents with low script purity and then deduplicated
using SHA-256 hashing at the document level. For Kokborok, raw sentences were identified
from publicly available corpora of Kokborok-language text; a unified public corpus of
20,300 Kokborok-language sentence pairs was compiled and filtered. Gold-standard datasets
were then constructed from these raw collections. Entity annotation was performed using a
structured human-in-the-loop pipeline in which candidate entity spans were first identified
algorithmically and subsequently verified or corrected by a human annotator. The final
decision for every accepted entity span was made by a human reviewer; machine-generated
suggestions served only as an initial draft layer.

The annotation schema adopts four entity categories standard in multilingual NER: PER
(person names), LOC (location and geopolitical entities), ORG (organizations and
institutions), and MISC (miscellaneous named entities not fitting the previous three
categories). Token-level labels follow the BIO scheme (Beginning, Inside, Outside),
yielding a nine-class label set (O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC,
I-MISC).

Each language corpus was partitioned into three non-overlapping splits: a training set, a
development set, and a Gold-Test set. The Gold-Test sets were kept isolated from model
development and were evaluated only after completion of the corresponding experimental
pipeline. Test-set integrity was protected using SHA-256 checksums, which were recorded
before each evaluation run and compared to the pre-computed reference hash to confirm
that the test file remained unmodified. The confirmed dataset statistics are presented in
Table 1.

---

**TABLE 1 (LaTeX)**

```latex
\begin{table}[htbp]
\centering
\caption{Dataset Characteristics and Gold-Test Distribution}
\label{tab:dataset}
\begin{tabular}{llllllll}
\hline
\textbf{Language} & \textbf{Script} & \textbf{Raw Data} & \textbf{Training} & \textbf{Development} & \textbf{Gold Test} & \textbf{Other Data} \\
\hline
Konkani  & Devanagari       & Wikipedia, news (gom wiki, Herald Goa, Bhaangarbhuin) & 850 sents. & 200 sents. & 200 sents. (124 entities) & — \\
Assamese & Assamese-Bengali & Wikipedia, news (as wiki, Pratidin, Niyomiya Barta)   & 850 sents. & 200 sents. & 200 sents. (138 entities) & — \\
Sinhala  & Sinhala Script   & Wikipedia, news (si wiki, Ada Derana, Dinamina)       & 850 sents. & 200 sents. & 200 sents. (132 entities) & — \\
Kokborok & Roman            & Public Kokborok corpus (20,300 sentence pairs)        & 849 sents. & 199 sents. & 199 sents. (16 entities)  & Exp. 10–11 public corpus \\
\hline
\end{tabular}
\end{table}
```

The verified Gold-Test entity counts reveal an important asymmetry in the Kokborok split:
the test set contains only 16 total annotated entity instances across 199 sentences,
compared to 124–138 entities in the corresponding splits of the other three languages.
This low entity density in the Kokborok test set means that each individual entity prediction
has a disproportionate effect on computed metrics, a factor that must be considered when
interpreting Kokborok performance figures. The test-set protection mechanism ensures that
no performance estimates reported in this study were informed by decisions made with
knowledge of the test-set contents.

---

## 6.2 Preprocessing and Grapheme-Level Tokenization

All raw text underwent a common preprocessing pipeline prior to feature extraction. The
pipeline was implemented from scratch without reliance on any pretrained tokenizer or
external language toolkit.

**Unicode Normalization.** All input text was canonically normalized to Unicode NFC form
using Python's `unicodedata.normalize` function. NFC normalization ensures that characters
represented by multiple equivalent Unicode code sequences—a frequent occurrence in
Devanagari, Assamese-Bengali, and Sinhala scripts—are reduced to a single canonical
composed form. In addition, byte order marks, zero-width spaces (`\u200b`), and redundant
invisible control characters were removed, while Zero-Width Joiners (ZWJ, `\u200d`) and
Zero-Width Non-Joiners (ZWNJ, `\u200c`) required for Indic and Sinhala conjunct formation
were preserved. A Sinhala-specific conjunct repair pass was applied before normalization to
restore corrupted ZWJ sequences common in digitized Sinhala text.

**Word Tokenization.** Sentences were segmented into word tokens using a Unicode-aware
rule-based tokenizer. The tokenizer matches sequences of Unicode letter, mark, number, and
joiner code points (`\p{L}\p{M}\p{N}\u200c\u200d`) as word tokens, and isolates punctuation
marks as separate tokens. This approach correctly preserves internal conjunct sequences in
Devanagari, Assamese, and Sinhala words, which would be corrupted by naive whitespace
splitting.

**Character-Level Grapheme Segmentation.** For character-level feature extraction, each
word token was further segmented into Unicode extended grapheme clusters using the `\X`
pattern of the Python `regex` library. This segmentation treats multi-codepoint grapheme
units—such as Devanagari consonant-vowel conjuncts and Sinhala consonant clusters—as
atomic character units rather than sequences of individual code points. This is particularly
important for Brahmic-script languages, where a single visible glyph may be composed of two
or more code points that jointly constitute a single phonological unit. Formally, given a
word token $w$, the grapheme sequence is defined as:

\[
\mathbf{c}(w) = (g_1, g_2, \ldots, g_{L_w})
\]

where each $g_i$ is an extended grapheme cluster and $L_w$ is the number of grapheme
clusters in the word.

**Language-Specific Handling.** The preprocessing pipeline applies script-specific
treatments conditioned on the source language. Sinhala text undergoes an additional
conjunct-repair pass to restore missing ZWJ characters before NFC normalization. Assamese
and Devanagari text receives standardization of the ASCII pipe character (`|`) to the
Devanagari danda (`।`) where used as a sentence terminator in digitized text.

---

## 6.3 Overall Research Framework

The research follows a sequential experimental pipeline proceeding from raw multilingual
data through model evaluation and explainability analysis. The pipeline comprises nine
stages:

1. Raw Data
2. Data Preparation
3. Unicode Normalization
4. Grapheme Tokenization
5. Feature Construction
6. Model Training
7. NER Prediction
8. Attribution
9. Faithfulness Evaluation

Raw multilingual text from each of the four languages is first processed through the
data-preparation and annotation pipeline to produce BIO-tagged CoNLL-format corpora.
These corpora then undergo Unicode NFC normalization and script-aware grapheme tokenization
as described in Section 6.2. The resulting token and character sequences serve as inputs
to feature construction, in which word-level, character-level, and language identity
embeddings are constructed from scratch. The combined feature representation is passed to
the proposed NER model for training. After training, the model produces BIO label predictions
over input sequences. The prediction stage is followed by an Attribution phase, in which
Gradient × Attention scores are computed for individual entity token positions. Finally, a
Faithfulness Evaluation is conducted separately using a token-erasure protocol to assess
the degree to which attribution scores align with model behavior under controlled input
perturbation.

---

**INSERT FIGURE 1 HERE**

Figure 1. Overall research framework of the proposed low-resource multilingual NER system,
showing nine sequential stages from raw data collection through faithfulness evaluation.

**GEMINI IMAGE PROMPT — FIGURE 1**

```
Create a professional, publication-quality research-paper workflow diagram suitable for an IEEE or Springer journal.

The diagram must show exactly NINE sequential stages connected by directional arrows. Each stage must be displayed as a clean, uniformly sized rectangular block with a short label inside.

The nine stages, in exact order from top to bottom (or left-to-right with a logical line break after stage 4 or 5), are:

1. Raw Data
2. Data Preparation
3. Unicode Normalization
4. Grapheme Tokenization
5. Feature Construction
6. Model Training
7. NER Prediction
8. Attribution
9. Faithfulness Evaluation

Stages 8 (Attribution) and 9 (Faithfulness Evaluation) MUST appear as two separate, distinct rectangular blocks connected by a single arrow. Do NOT merge them into one block.

Use a clean white background.
Use dark navy or dark charcoal rectangular blocks.
Use white or near-white sans-serif text labels inside each block.
Use thin, directional arrows connecting each stage to the next.
Typography must be clean and readable at journal paper scale (10–11pt equivalent).
Consistent block sizes and spacing throughout.
Do NOT show any neural network internal architecture components (no LSTM cells, no CNN diagrams, no attention heads).
Do NOT include any performance numbers, accuracy values, F1 scores, or model checkpoint names.
Do NOT include any decorative icons, logos, gradients, or marketing-style visual elements.
All nine stages must be labeled with exactly the text listed above — do not paraphrase or rename any stage.
High resolution, suitable for print publication.
```

---

## 6.4 Proposed Model Architecture

The proposed model is a multilingual, language-conditioned sequence-labeling architecture
designed for low-resource NER. All components are trained from scratch without utilizing
any pretrained embeddings, pretrained tokenizers, or pretrained model checkpoints.

### 6.4.1 Composite Input Representation

For each token position $t$ in an input sentence, the model constructs a composite
representation by concatenating three distinct feature vectors:

1. **Word embedding** $\mathbf{e}^{(w)}_t \in \mathbb{R}^{100}$: a randomly initialized,
   learned dense vector indexed by the word-level vocabulary.
2. **Character-level representation** $\mathbf{e}^{(c)}_t \in \mathbb{R}^{90}$: a
   morphological feature vector produced by the character-level CNN encoder (described in
   Section 6.4.2) from the grapheme sequence of the word at position $t$.
3. **Language embedding** $\mathbf{e}^{(\ell)}_t \in \mathbb{R}^{16}$: a learned vector
   representing the source language identity, broadcast uniformly across all token positions
   within a sentence.

The three vectors are concatenated to form a 206-dimensional fused representation:

\[
\mathbf{x}_t = [\mathbf{e}^{(w)}_t \;\|\; \mathbf{e}^{(c)}_t \;\|\; \mathbf{e}^{(\ell)}_t]
\in \mathbb{R}^{206}
\]

The dimensions sum to 206 (100 + 90 + 16), verified from both the model docstring and the
per-language configuration files. A dropout layer with rate 0.3 is applied to the
concatenated representation before it is passed to the sequence encoder.

The scientific motivation for this composite representation is threefold. The word embedding
provides a lexical identity signal. The character-level representation provides morphological
and subword information that can partially generalize to rare and unseen word forms. The
language embedding allows the shared encoder to condition its processing on the source
language, supporting cross-lingual parameter sharing while maintaining language-specific
representational distinctions.

### 6.4.2 Character-Level CNN

The character-level CNN encoder extracts morphological features from the grapheme-cluster
sequence of each word. Given the grapheme sequence $\mathbf{c}(w) = (g_1, \ldots, g_{L_w})$,
each grapheme cluster $g_i$ is mapped to a learned embedding vector $\mathbf{e}^{(g)}_i
\in \mathbb{R}^{50}$ via a randomly initialized character embedding table. A dropout of
rate 0.3 is applied to these embeddings.

Three parallel one-dimensional convolutional filters are then applied over the embedded
grapheme sequence with kernel widths $k \in \{3, 4, 5\}$ and 30 filters per width. For
each filter width $k$, the convolution is followed by a ReLU nonlinearity and global
max-over-time pooling:

\[
p^{(k)} = \max_{i} \; \text{ReLU}\!\left(\mathbf{W}^{(k)} * \mathbf{E}^{(g)} + \mathbf{b}^{(k)}\right)
\]

The three pooled vectors are concatenated to form the 90-dimensional character-level
word representation $\mathbf{e}^{(c)}_t \in \mathbb{R}^{90}$.

### 6.4.3 BiLSTM Context Encoder

The sequence of 206-dimensional token representations $(\mathbf{x}_1, \ldots, \mathbf{x}_T)$
is passed to a two-layer bidirectional LSTM, each direction having a hidden size of 128.
The forward hidden state $\overrightarrow{\mathbf{h}}_t$ and the backward hidden state
$\overleftarrow{\mathbf{h}}_t$ are concatenated at each position to yield a 256-dimensional
contextual representation:

\[
\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t \;\|\; \overleftarrow{\mathbf{h}}_t]
\in \mathbb{R}^{256}
\]

An output dropout of rate 0.3 is applied to the BiLSTM hidden states.

### 6.4.4 Multi-Head Self-Attention

The BiLSTM hidden states $\mathbf{H} = (\mathbf{h}_1, \ldots, \mathbf{h}_T) \in
\mathbb{R}^{T \times 256}$ are passed through a single block of multi-head scaled
dot-product self-attention with four attention heads. The 256-dimensional hidden space
is divided equally across the four heads, yielding a per-head dimension of $d_k = 64$.

For each head, query, key, and value projections are computed:

\[
\mathbf{Q} = \mathbf{H}\mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{H}\mathbf{W}_K, \quad
\mathbf{V} = \mathbf{H}\mathbf{W}_V
\]

where $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{256 \times 64}$ per head.
The scaled dot-product attention weights are:

\[
\mathbf{A} = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)
\]

where padding positions are masked to $-10^9$ before the softmax. The context vector is:

\[
\text{Context} = \mathbf{A}\mathbf{V}
\]

The four head outputs are concatenated and projected through an output layer
$\mathbf{W}_O \in \mathbb{R}^{256 \times 256}$. A residual connection and Layer
Normalization are then applied:

\[
\tilde{\mathbf{H}} = \text{LayerNorm}(\mathbf{H} + \text{Dropout}(\text{Attn}(\mathbf{H})))
\]

Both the residual connection and the Layer Normalization are verified from the implementation
(`self_attention.py`, line 118). The attention weights $\mathbf{A}^{(h)}$ for all four
heads are stored after each forward pass for subsequent attribution analysis. Dropout at
rate 0.1 is applied within the attention computation.

### 6.4.5 CRF Decoder

The attended representations $\tilde{\mathbf{H}} \in \mathbb{R}^{T \times 256}$ are
projected to tag emission logits through a **shared** linear layer and decoded by a
**shared** linear-chain Conditional Random Field (CRF). The shared configuration—verified
from all four per-language YAML configuration files (`crf_config: shared`)—uses a single
emission projection and a single CRF transition matrix across all four languages, enabling
maximal parameter sharing during training.

The CRF models the joint probability of the complete label sequence
$\mathbf{y} = (y_1, \ldots, y_T)$:

\[
P(\mathbf{y} \mid \mathbf{x}) = \frac{\exp(S(\mathbf{x}, \mathbf{y}))}{Z(\mathbf{x})}
\]

where:

\[
S(\mathbf{x}, \mathbf{y}) = \sum_{t=1}^{T} \phi(y_t, t) + \sum_{t=1}^{T-1} \psi(y_t, y_{t+1})
\]

with $\phi(y_t, t)$ the emission score for tag $y_t$ at position $t$, and $\psi(y_t, y_{t+1})$
the learned transition score between consecutive labels. Training minimizes the negative
log-likelihood. Inference employs the Viterbi algorithm to decode the globally optimal
label sequence, enforcing BIO-consistent tag transitions at the sequence level.

### 6.4.6 Explainability Branch

The model's attention module is instrumented to expose the per-head attention weight
tensors $\mathbf{A}^{(h)} \in \mathbb{R}^{T \times T}$ for all $h \in \{1, 2, 3, 4\}$
after each forward pass. These tensors serve as inputs to the attribution branch, which
computes token attributions post-hoc from the frozen model. The attribution branch does
not modify any model parameters and adds no trainable components to the architecture.
The conceptual relationship is:

NER model (frozen) → attention weight tensors → Gradient × Attention computation → token-level attribution scores

Attention weights alone do not constitute an explanation of model decisions; they are
intermediate representations that, when combined with gradient information in the
attribution computation, produce scores that can be evaluated empirically through the
faithfulness protocol.

---

**INSERT FIGURE 2 HERE**

Figure 2. Detailed architecture of the proposed explainable NER model, showing the
composite 206-dimensional input representation, BiLSTM encoder, four-head self-attention
with residual connection and Layer Normalization, shared CRF decoder, and post-hoc
attribution branch.

**GEMINI IMAGE PROMPT — FIGURE 2**

```
Create a professional, publication-quality neural architecture diagram for an IEEE or Springer research paper.

Layout: top to bottom.

At the very top, show THREE separate input source boxes arranged side by side:
- Left box: "Word Tokens"
- Middle box: "Character Grapheme Sequences"
- Right box: "Language ID"

Below the left box, draw a rectangular block labeled:
  "Word Embedding (100d, randomly initialized)"

Below the middle box, draw a rectangular block labeled:
  "Char CNN
   (char embed 50d → conv widths 3,4,5 × 30 filters → 90d, ReLU + max-pool)"

Below the right box, draw a rectangular block labeled:
  "Language Embedding (16d, learned)"

Draw arrows from these three blocks converging into a single wide rectangular block labeled:
  "Concatenation + Dropout (0.3) → 206d"

Below that, draw a rectangular block labeled:
  "BiLSTM Encoder (2 layers, hidden 128 each direction, output 256d, Dropout 0.3)"

Below that, draw a rectangular block labeled:
  "Multi-Head Self-Attention
   (4 heads, d_k = 64, 1 block, Dropout 0.1)
   + Residual Connection + LayerNorm → 256d"

From the right side of the Multi-Head Self-Attention block, draw a side-branch arrow pointing to a box labeled:
  "Attribution Branch
   (Gradient × Attention, post-hoc, no trainable parameters)"

Below the Multi-Head Self-Attention block, draw a rectangular block labeled:
  "Linear Emission Projection (256d → 9 tags, shared)"

Below that, draw a rectangular block labeled:
  "Shared Linear-Chain CRF (Viterbi decoding)"

At the very bottom, draw a rectangular block labeled:
  "NER Output (BIO label sequence)"

Use a clean white background.
Use dark navy or dark charcoal rectangular blocks.
Use white or near-white sans-serif text labels inside each block.
Use thin directional arrows to show data flow throughout.
Label the key dimensionalities (100d, 90d, 16d, 206d, 256d) beside the relevant connections.
Do NOT show "per-language" or "shared or per-language" — the CRF is shared and must be labeled as shared.
Do NOT include any pretrained model names (no BERT, XLM-R, mBERT, etc.).
Do NOT include any performance numbers or F1 scores.
Do NOT include any modules not listed above.
Do NOT add any decorative icons, background shading, or gradients.
Typography must be clean, consistent, and readable at journal-paper scale.
High resolution, publication quality.
```

The architecture integrates word-level lexical information, character-level morphological
features, and language identity conditioning within a unified 206-dimensional input
representation. The BiLSTM provides full-sentence contextual encoding, and the four-head
self-attention applies a global relational operation over those states, enabling each token
to attend to any other regardless of distance. The shared CRF decoder enforces structured
BIO-consistent label sequences across all four languages within a single parametric model.
This design is appropriate for low-resource multilingual NER because: (i) the character CNN
provides subword morphological generalization that partially compensates for sparse lexical
coverage; (ii) language conditioning allows the shared backbone to distinguish cross-lingual
signals without duplicating parameters; and (iii) the CRF enforces label consistency that
per-token softmax classification cannot guarantee.

---

## TABLE 2 (LaTeX)

```latex
\begin{table}[htbp]
\centering
\caption{Layer-Wise Architecture and Parameter Specifications}
\label{tab:architecture}
\begin{tabular}{lllll}
\hline
\textbf{Layer / Component} & \textbf{Input Dim.} & \textbf{Output Dim.} & \textbf{Configuration} & \textbf{Notes} \\
\hline
Word Embedding         & —     & 100d  & Randomly init., from scratch & Dropout 0.3 \\
Char Grapheme Embedding & —    & 50d   & Randomly init., from scratch & Per extended grapheme cluster \\
Char CNN Convolutions  & 50d   & 90d   & Widths \{3,4,5\}, 30 filters each & ReLU + max-over-time pool \\
Language Embedding     & —     & 16d   & 4 languages, from scratch    & Dropout 0.1 \\
Concatenation + Dropout & —    & 206d  & 100 + 90 + 16                & Dropout 0.3 \\
BiLSTM Encoder         & 206d  & 256d  & 2 layers, hidden=128/dir., bidirectional & Dropout 0.3 \\
Multi-Head Self-Attn   & 256d  & 256d  & 4 heads, $d_k=64$, 1 block   & Residual + LayerNorm, Dropout 0.1 \\
Linear Emission Proj.  & 256d  & 9d (tags) & Shared across languages  & Xavier init. \\
Shared Linear-Chain CRF & 9d emissions & — & BIO label set (9 tags), shared & Viterbi decode at inference \\
\hline
\end{tabular}
\end{table}
```

---

## 6.5 Evaluated Model Configurations

The project implements four progressively developed model configurations representing an
incremental architectural progression from a minimal word-level baseline to the full
proposed multilingual system. All four configurations are present in the project codebase
under `model/`. The architectural characteristics of each configuration are described
below. Quantitative evaluation evidence is available for the final multilingual
configuration (Model 4); the earlier configurations are implemented and architecturally
documented, but independent evaluation logs confirming Gold-Dev or Gold-Test results for
each configuration individually are not present in the project artifacts and are therefore
not claimed here.

### 6.5.1 Model 1 — Word-Level BiLSTM-CRF Baseline

The first configuration is a word-level BiLSTM-CRF model (`ner_model_v1_baseline.py`,
Phase 5). It consists of a randomly initialized word embedding of dimension 100, followed
by a single-layer bidirectional LSTM with hidden size 128, a linear emission
projection, and a linear-chain CRF decoder. Neither character-level features nor language
conditioning are included. This configuration represents the minimal sequence-labeling
architecture, establishing a lexical-only lower bound against which character and attention
augmentations can be compared.

### 6.5.2 Model 2 — Char-CNN + BiLSTM-CRF

The second configuration (`ner_model_v2_charcnn.py`, Phase 6) augments the word-level
baseline with the character-level CNN encoder. The concatenated word (100d) and character
(90d) representations form a 190-dimensional fused input to a two-layer bidirectional
LSTM. A CRF decoder is used for sequence labeling. No language embedding or self-attention
is included. This configuration isolates the architectural contribution of character-level
morphological features.

### 6.5.3 Model 3 — Char-CNN + BiLSTM + Self-Attention + CRF

The third configuration (`ner_model_v3_attention.py`, Phase 7) adds the multi-head
self-attention block to Model 2. The BiLSTM output (256d) passes through the four-head
attention block, followed by a residual connection and Layer Normalization, before the
linear emission projection and CRF decoder. No language embedding is included. This
configuration constitutes the minimal architecture from which attention-based attribution
can be extracted, and provides the foundation for the explainability analysis.

### 6.5.4 Model 4 — Final Proposed Multilingual Language-Conditioned Model

The fourth and final configuration (`ner_model_v4_multilingual.py`, Phase 8) is the
complete proposed model described in Section 6.4. It adds the 16-dimensional language
conditioning embedding, forming the 206-dimensional composite input. The CRF head is
configured as shared (`crf_config: shared`), verified from all four per-language
configuration files. This is the only configuration for which complete Gold-Test evaluation
evidence exists in the project artifacts, and it constitutes the primary subject of
evaluation.

---

## TABLE 3 (LaTeX)

```latex
\begin{table}[htbp]
\centering
\caption{Evaluated Model Configurations}
\label{tab:models}
\begin{tabular}{llll}
\hline
\textbf{Model} & \textbf{Main Components} & \textbf{Additional Component} & \textbf{Purpose} \\
\hline
Model 1 (v1) & Word Embed (100d) + BiLSTM + CRF          & —                     & Lexical-only baseline \\
Model 2 (v2) & Word Embed (100d) + Char-CNN (90d) + BiLSTM + CRF & —           & +Character morphology \\
Model 3 (v3) & Word Embed (100d) + Char-CNN (90d) + BiLSTM + Attn + CRF & —    & +Self-attention + Attribution \\
Model 4 (v4) & Word (100d) + Char (90d) + Lang (16d) → 206d + BiLSTM + Attn + Shared CRF & Language Embedding & +Language conditioning (final) \\
\hline
\multicolumn{4}{l}{\small Note: Gold-Test evaluation evidence is available for Model 4 only. Earlier configurations are architecturally implemented.} \\
\hline
\end{tabular}
\end{table}
```

---

## 6.6 Mathematical Formulation

### 6.6.1 Input and Composite Representation

Let $w_t$ denote the word token at position $t$, $\mathbf{c}(w_t)$ its grapheme-cluster
sequence, and $\ell$ the integer language identifier. The composite input representation is:

\[
\mathbf{x}_t = [\mathbf{e}^{(w)}(w_t) \;\|\; \text{CharCNN}(\mathbf{c}(w_t)) \;\|\; \mathbf{e}^{(\ell)}(\ell)]
\in \mathbb{R}^{206}
\]

### 6.6.2 Character-Level Representation

Given the grapheme embedding matrix $\mathbf{E}^{(g)} \in \mathbb{R}^{L_w \times 50}$,
the CNN applies convolutions for each filter width $k \in \{3, 4, 5\}$:

\[
\mathbf{f}^{(k)}_i = \text{ReLU}\!\left(\mathbf{W}^{(k)} \cdot \mathbf{E}^{(g)}_{i:i+k-1} + b^{(k)}\right)
\]

Global max-over-time pooling extracts the most prominent feature:

\[
p^{(k)} = \max_{i} \; \mathbf{f}^{(k)}_i \in \mathbb{R}^{30}
\]

\[
\text{CharCNN}(\mathbf{c}(w)) = [p^{(3)} \;\|\; p^{(4)} \;\|\; p^{(5)}] \in \mathbb{R}^{90}
\]

### 6.6.3 BiLSTM Context Encoding

\[
\overrightarrow{\mathbf{h}}_t = \text{LSTM}_{\rightarrow}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1}),
\quad
\overleftarrow{\mathbf{h}}_t = \text{LSTM}_{\leftarrow}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})
\]

\[
\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t \;\|\; \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{256}
\]

### 6.6.4 Multi-Head Self-Attention

For each head $h$ with $d_k = 64$:

\[
\mathbf{A}^{(h)} = \text{softmax}\!\left(\frac{\mathbf{H}\mathbf{W}^{(h)}_Q \left(\mathbf{H}\mathbf{W}^{(h)}_K\right)^\top}{\sqrt{d_k}}\right),
\quad
\text{Head}^{(h)} = \mathbf{A}^{(h)} \mathbf{H}\mathbf{W}^{(h)}_V
\]

\[
\text{MultiHead}(\mathbf{H}) = [\text{Head}^{(1)} \;\|\; \cdots \;\|\; \text{Head}^{(4)}]\, \mathbf{W}_O
\in \mathbb{R}^{T \times 256}
\]

\[
\tilde{\mathbf{H}} = \text{LayerNorm}\!\left(\mathbf{H} + \text{Dropout}\!\left(\text{MultiHead}(\mathbf{H})\right)\right)
\]

### 6.6.5 CRF Sequence Scoring and Viterbi Decoding

\[
S(\mathbf{x}, \mathbf{y}) = \sum_{t=1}^{T} \phi(y_t, t) + \sum_{t=1}^{T-1} \psi(y_t, y_{t+1})
\]

\[
\mathcal{L} = -S(\mathbf{x}, \mathbf{y}^*) + \log Z(\mathbf{x}), \quad
\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} \; S(\mathbf{x}, \mathbf{y})
\]

### 6.6.6 Evaluation Metrics

\[
\text{Precision} = \frac{TP}{TP + FP}, \quad
\text{Recall} = \frac{TP}{TP + FN}, \quad
F_1 = \frac{2 \cdot P \cdot R}{P + R}
\]

\[
\text{Token Accuracy} = \frac{\text{Correctly labeled tokens}}{\text{Total tokens}}
\]

Entity-level F1 is the primary metric (exact span boundary and type match). Token accuracy
is reported as a secondary metric.

### 6.6.7 Attribution Formulation

For a target entity token at query position $q$ with predicted tag emission score
$s_{q,y^*}$, the Gradient × Attention attribution of key position $j$ in head $h$ is:

\[
\alpha^{(h)}_{q \to j} = \frac{\partial s_{q,y^*}}{\partial \mathbf{A}^{(h)}_{q,j}} \cdot \mathbf{A}^{(h)}_{q,j}
\]

Aggregated across the four heads by mean:

\[
\alpha_{q \to j} = \frac{1}{4} \sum_{h=1}^{4} \alpha^{(h)}_{q \to j}
\]

The implementation applies absolute-value normalization to obtain a magnitude-based,
scale-invariant attribution score:

\[
\hat{\alpha}_{q \to j} = \frac{|\alpha_{q \to j}|}{\sum_{j'} |\alpha_{q \to j'}|}
\]

This formulation is verified directly from `explainability/gradient_attention_attribution.py`
(lines 147–159).

### 6.6.8 Faithfulness Formulation

For each entity prediction, the top-$k$ tokens by attribution score are identified and
replaced with the `<UNK>` token. The model is re-evaluated on the perturbed input and the
drop in entity emission confidence is measured. A random-$k$ erasure baseline replaces $k$
randomly selected non-top tokens. The faithfulness gap for attribution method $m$ at
erasure size $k$ is:

\[
\Delta_k^{(m)} = \overline{\Delta}_{\text{top-}k}^{(m)} - \overline{\Delta}_{\text{rand-}k}
\]

where $\overline{\Delta}$ denotes the mean confidence drop over the evaluation sample. A
positive gap $\Delta_k^{(m)} > 0$ indicates that erasing the top-attributed tokens reduces
model confidence more than random erasure, providing empirical evidence under this specific
protocol that the attribution scores reflect information relevant to the prediction.

---

## 6.7 Multi-Stage Training and Experimental Procedure

Training followed a multi-stage procedure designed to transfer shared cross-lingual
representations before language-specific specialization.

**Stage 1 — Multilingual Joint Training (Verified).** The shared model backbone (word
embedding, character CNN, language embedding, BiLSTM encoder, self-attention block, shared
CRF head) was jointly trained on the combined multilingual training corpus. This joint
training allows the encoder to develop generalizable representations across all four
languages simultaneously, and the resulting joint checkpoint serves as both the
initialization for per-language fine-tuning and the reference point for joint-baseline
performance comparison.

**Stage 2 — Layer-Wise Unfreezing Specialization (Verified).** Following joint training,
each language was fine-tuned independently using a phased layer-unfreezing schedule
(`training/layer_unfreeze_schedule.py`). In the phased schedule, lower layers (word
embedding, character CNN, language embedding, BiLSTM encoder) are frozen during early
epochs (epochs 1 to `unfreeze_epoch`$-$1, where `unfreeze_epoch = 4` per configuration),
while upper layers (self-attention, emission projection, CRF head) are updated from the
beginning. All layers are unfrozen from `unfreeze_epoch` onward. This strategy preserves
shared cross-lingual representations in the lower layers during the initial fine-tuning
phase while allowing higher-level layers to specialize.

The configuration files further confirm the use of the AdamW optimizer with a learning rate
of 0.0002, weight decay of 1e-4, gradient norm clipping of 5.0, and a batch size of 32
during the fine-tuning stage. These values are included in Table 4 below.

No evidence of completed confidence-filtered self-training, targeted entity substitution
augmentation, or a bounded hyperparameter search was found in the project artifacts.
These procedures are documented in configuration definitions but are not claimed as
completed experimental procedures.

---

## TABLE 4 (LaTeX)

```latex
\begin{table}[htbp]
\centering
\caption{Verified Training and Fine-Tuning Configuration (Per-Language Final Stage)}
\label{tab:training}
\begin{tabular}{ll}
\hline
\textbf{Parameter / Strategy} & \textbf{Setting} \\
\hline
Optimizer                & AdamW \\
Learning rate (fine-tune) & 0.0002 \\
Weight decay             & 1e-4 \\
Gradient norm clipping   & 5.0 \\
Batch size               & 32 \\
Unfreezing schedule      & Phased (lower layers frozen for first 3 epochs) \\
Lower layers frozen      & Word embedding, Char CNN, Language embedding, BiLSTM \\
Upper layers updated from epoch 1 & Self-attention, linear projection, CRF \\
CRF configuration        & Shared \\
\hline
\end{tabular}
\end{table}
```

---

## 6.8 Explainability and Faithfulness Method

The explainability component of this study addresses the question of whether the model's
internal attention representations contain information that is consistent with its NER
predictions. A fundamental conceptual distinction is maintained throughout:

> Attention visualization, gradient-based attribution, and faithfulness evaluation constitute
> three distinct operations that address different questions about model behavior.

**Attention visualization** records the raw attention weight distributions $\mathbf{A}^{(h)}$
produced during a forward pass. These weights indicate how much attention each position
receives from each query position, but they do not, by themselves, establish which tokens
are causally important to a given prediction.

**Gradient × Attention attribution** combines the attention weights with gradient information
to produce a token-level attribution score. As formalized in Section 6.6.7, the method
computes the element-wise product of the gradient of the target emission score with respect
to the attention weights, and the attention weights themselves. This joint quantity reflects
both the magnitude of the attention signal and the sensitivity of the prediction to changes
in that signal. The implementation uses direct PyTorch autograd without any external
interpretability library (`explainability/gradient_attention_attribution.py`). Attribution
scores are averaged across the four attention heads and normalized to unit L1 sum after
taking absolute values to yield a magnitude-based, scale-invariant score per token.

The same pipeline also returns raw attention scores (mean of $\mathbf{A}^{(h)}_{q,:}$
across heads) for comparison with Gradient × Attention under the faithfulness protocol.

**Attribution visualization** produces token-level heatmaps over input sentences.
For each correctly predicted entity, token-attribution scores are represented as a
color-coded overlay. Heatmaps illustrate which tokens receive the largest attribution
scores for a given prediction; they do not constitute proof that the highlighted tokens
causally determined the prediction, and they are not used as primary evidence of model
behavior.

**Faithfulness evaluation** is conducted separately using the token-erasure protocol
defined in Section 6.6.8, implemented in `explainability/erasure_faithfulness_test.py`.
By measuring the empirical confidence drop when top-attributed tokens are erased relative
to a random-erasure baseline, the evaluation provides an experimental test of whether
attribution scores correlate with model-relevant information. This evaluation compares
Raw Attention and Gradient × Attention on samples of 100 gold-development entity
predictions per language for $k \in \{1, 2, 3\}$. Faithfulness gap and win-rate results
are reported in Section 7.

Attribution weights provide an internal signal that can be visualized, whereas the Gradient
× Attention method combines model sensitivity with attention information. Faithfulness is
then assessed independently by testing whether perturbing or erasing highly attributed
information produces corresponding changes in model behavior. Attribution scores, even when
faithfulness gaps are positive, do not constitute causal evidence: they are an approximation
whose interpretive scope is bounded by the assumptions of the erasure-based evaluation
protocol.

---

# EVIDENCE AND VERIFICATION NOTES

The following items requested in the methodology specification remain without artifact support:

1. **Exact parameter counts** — no experiment log or script records total parameter counts
   per layer. Table 2 omits parameter counts.
2. **Confidence-filtered self-training** — no saved self-training outputs, pseudo-label
   files, or self-training checkpoints found. Excluded from the manuscript.
3. **Targeted entity substitution augmentation** — no augmented corpus artifact found.
   Excluded.
4. **Bounded hyperparameter search** — documented in `base_config.yaml` under `[search]`
   but no sweep log confirming completed runs found. Not claimed as a completed experiment.
5. **Gold-Dev/Test evaluation logs for Models 1, 2, and 3** — the model files are verified
   (`ner_model_v1_baseline.py`, `v2`, `v3`) but no standalone evaluation JSON for these
   configurations is present in `reports/experiment_logs/`. Table 3 notes this explicitly.
6. **Integrated Gradients** — no implementation artifact found. Not included.
7. **ReLU Grad × Attention, Grad × Input** — not present in `phase15_faithfulness_four_language_raw.json`. Not included.

**CRF Configuration (verified):** All four per-language final configuration files
(`configs/final_konkani.yaml`, `final_assamese.yaml`, `final_sinhala.yaml`,
`final_kokborok.yaml`) set `crf_config: shared`. Section 6.4.5, Table 2, Table 3, and the
Figure 2 prompt have been updated to reflect this.

**Raw Data (partially verified):** The `base_config.yaml` documents Wikipedia and news
sources for Konkani, Assamese, and Sinhala. The Kokborok raw data is documented as a
unified public corpus of 20,300 sentence pairs confirmed from
`data_collection/unified_public_corpus.json`. Precise article counts for each language
are not separately recorded in a verified artifact and are therefore described qualitatively
in Table 1.

No training was performed during the preparation of this manuscript section.
No Gold-Test evaluation was conducted.
No model checkpoints or datasets were modified.
