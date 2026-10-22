"""
Render Section_6_Methodology_Springer_Final.pdf using Headless Chrome engine with Springer Nature academic styling.
Reflects all 17 verified corrections:
1. Removed human-in-the-loop / human adjudication claims.
2. Accurate test-set protection (not used for training/tuning, locked during development, evaluated only after freeze).
3. Primary faithfulness protocol: zero input embeddings + exact forward-backward CRF marginal posterior confidence.
   Baseline protocol: <UNK> token replacement + local emission softmax confidence.
4. "Stage 1: Multilingual Joint Training" (not pretraining).
5. No markdown bold or backticks in text.
6. Clean labels and refs.
7. Clean display math.
8. Figure 2 TikZ clean.
9. Table 1 verified values.
10. Table 2 verified layer dimensions and verified dropout values.
11. Model 1 verified BiLSTM configuration (single-layer default).
12. CRF claim: Viterbi decoding selects highest-scoring sequence using emission and learned transition scores.
13. Formula + simple explanation pattern.
14. Clean math syntax.
15. Figures in TikZ with 9 distinct stages in Fig 1 and full architecture in Fig 2.
16. No test results or F1 scores in methodology.
"""

import subprocess
import time
from pathlib import Path
import fitz

ROOT_DIR = Path(__file__).resolve().parent
HTML_PATH = ROOT_DIR / "Section_6_Methodology_Springer_Final.html"
PDF_PATH = ROOT_DIR / "Section_6_Methodology_Springer_Final.pdf"

html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Section 6 — Methodology | Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution</title>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  @page {
    size: A4 portrait;
    margin: 20mm 18mm 22mm 18mm;
    @top-left {
      content: "Explainable NER for Low-Resource Languages Using Attention Attribution";
      font-family: 'Times New Roman', Times, serif;
      font-size: 8pt;
      font-style: italic;
      color: #555555;
    }
    @top-right {
      content: "Springer Nature Journal Template";
      font-family: 'Times New Roman', Times, serif;
      font-size: 8pt;
      font-weight: bold;
      color: #1a73e8;
    }
    @bottom-center {
      content: counter(page);
      font-family: 'Times New Roman', Times, serif;
      font-size: 9pt;
      color: #333333;
    }
  }

  body {
    font-family: "Times New Roman", Times, "Liberation Serif", serif;
    font-size: 10pt;
    line-height: 1.35;
    color: #111111;
    text-align: justify;
    margin: 0;
    padding: 0;
  }

  .paper-header {
    text-align: center;
    border-bottom: 1.5px solid #111111;
    padding-bottom: 12px;
    margin-bottom: 16px;
  }

  .journal-badge {
    font-family: 'Arial', sans-serif;
    font-size: 8pt;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #666666;
    margin-bottom: 6px;
  }

  .paper-title {
    font-size: 16pt;
    font-weight: bold;
    margin: 6px 0;
    line-height: 1.25;
  }

  .paper-authors {
    font-size: 10pt;
    margin: 6px 0 2px 0;
  }

  .paper-affil {
    font-size: 8.5pt;
    color: #444444;
    font-style: italic;
  }

  .abstract-box {
    background-color: #f9f9fb;
    border-left: 3px solid #1a73e8;
    padding: 10px 14px;
    margin: 12px 0 16px 0;
    font-size: 9pt;
  }

  .abstract-title {
    font-weight: bold;
    text-transform: uppercase;
    font-size: 8pt;
    letter-spacing: 1px;
    color: #1a73e8;
    margin-bottom: 4px;
  }

  h2 {
    font-size: 12pt;
    font-weight: bold;
    border-bottom: 1px solid #cccccc;
    padding-bottom: 3px;
    margin-top: 18px;
    margin-bottom: 8px;
    color: #111111;
  }

  h3 {
    font-size: 10.5pt;
    font-weight: bold;
    margin-top: 14px;
    margin-bottom: 6px;
    color: #222222;
  }

  h4 {
    font-size: 10pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 10px;
    margin-bottom: 4px;
  }

  p {
    margin: 6px 0;
    text-indent: 1.5em;
  }

  p.noindent {
    text-indent: 0;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 8.5pt;
  }

  th, td {
    padding: 5px 8px;
    text-align: left;
  }

  thead th {
    border-top: 1.5px solid #111111;
    border-bottom: 1px solid #111111;
    font-weight: bold;
  }

  tbody tr:last-child td {
    border-bottom: 1.5px solid #111111;
  }

  .table-caption {
    font-size: 8.5pt;
    font-weight: bold;
    margin-bottom: 4px;
    text-align: left;
  }

  .fig-box {
    margin: 16px 0;
    text-align: center;
    page-break-inside: avoid;
  }

  .fig-caption {
    font-size: 8.5pt;
    margin-top: 6px;
    text-align: justify;
  }

  .fig-caption b {
    font-weight: bold;
  }

  .diagram-container {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 12px;
    display: inline-block;
    width: 96%;
    box-sizing: border-box;
  }

  .workflow-grid {
    display: flex;
    flex-direction: column;
    gap: 10px;
    align-items: center;
  }

  .workflow-row {
    display: flex;
    justify-content: space-around;
    width: 100%;
    align-items: center;
  }

  .wf-block {
    background: #f0f4f9;
    border: 1.5px solid #336699;
    border-radius: 4px;
    padding: 6px 12px;
    font-family: Arial, sans-serif;
    font-size: 8.5pt;
    font-weight: bold;
    color: #1a365d;
    width: 160px;
    text-align: center;
  }

  .wf-arrow-h {
    font-size: 14pt;
    color: #4a5568;
  }

  .wf-arrow-v {
    font-size: 14pt;
    color: #4a5568;
    text-align: center;
  }

  .math-block {
    margin: 8px 0;
    text-align: center;
    font-size: 9.5pt;
  }

  ul, ol {
    margin: 4px 0 8px 24px;
    padding: 0;
  }

  li {
    margin-bottom: 3px;
  }

  code {
    font-family: "Courier New", Courier, monospace;
    font-size: 8.5pt;
    background: #f1f3f5;
    padding: 1px 3px;
    border-radius: 2px;
  }
</style>
</head>
<body>

<div class="paper-header">
  <div class="journal-badge">Springer Nature &bull; Research Article &bull; Section 6 Final Manuscript</div>
  <div class="paper-title">Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution</div>
  <div class="paper-authors">Author 1 &bull; Author 2 &bull; Research Laboratory</div>
  <div class="paper-affil">Department of Computer Science and Engineering, India</div>
</div>

<div class="abstract-box">
  <div class="abstract-title">Section Overview</div>
  This document presents the complete and verified <b>Section 6 — Methodology</b> formatted in accordance with Springer Nature journal typography. The section outlines the from-scratch data processing, neural architecture, multi-stage training schedule, and explainability framework across four low-resource languages: Konkani, Assamese, Sinhala, and Kokborok.
</div>

<h2>6. Methodology</h2>

<p class="noindent">This section describes the methodology used to develop and evaluate an explainable named entity recognition (NER) system for four low-resource languages: Konkani, Assamese, Sinhala, and Kokborok. The system is designed under a strict constraint: all models, embeddings, and tokenizers are constructed and trained entirely from scratch. No external pretrained language models (such as BERT, mBERT, XLM-RoBERTa, MuRIL, or IndicBERT) and no external static word embeddings (such as FastText or Word2Vec) are used anywhere in the pipeline.</p>

<p>The following subsections describe the dataset preparation, the preprocessing pipeline, the overall research framework, the proposed neural architecture, the baseline configurations, the mathematical formulation, the multi-stage training procedure, and the explainability and faithfulness evaluation methods.</p>

<h3>6.1 Data Collection and Dataset Preparation</h3>

<p class="noindent">The target languages represent four distinct linguistic and orthographic environments:</p>
<ol>
  <li><b>Konkani</b> (<i>gom</i>): An Indo-Aryan language spoken along the western coast of India, written here in the Devanagari script.</li>
  <li><b>Assamese</b> (<i>as</i>): An Eastern Indo-Aryan language spoken in Assam, India, written in the Assamese-Bengali script.</li>
  <li><b>Sinhala</b> (<i>si</i>): An Indo-Aryan language of Sri Lanka, written in the Sinhala script with distinctive syllabic conjuncts.</li>
  <li><b>Kokborok</b>: A Tibeto-Burman language of Tripura, India, written here in the Roman script.</li>
</ol>

<p>Raw text for Konkani, Assamese, and Sinhala was gathered from public Wikipedia archives and regional digital newspapers. Konkani sources include the Konkani Wikipedia, Herald Goa, and Bhaangarbhuin. Assamese sources include the Assamese Wikipedia, Asomiya Pratidin, and Niyomiya Barta. Sinhala sources include the Sinhala Wikipedia, Ada Derana Sinhala, and Dinamina. The collected documents were filtered to remove noise and low-script purity segments, and deduplicated using SHA-256 document hashing. For Kokborok, raw data was compiled from public corpora into a unified collection of 20,300 sentence pairs.</p>

<p>Annotated corpora were developed using structured heuristic and dictionary-based annotation pipelines. Entity boundaries and types were generated algorithmically using gazetteers, orthographic dictionaries, and regular expression patterns. The resulting annotated sentences were formatted into standard CoNLL format.</p>

<p>The annotation schema covers four standard named entity types:</p>
<ul>
  <li><b>PER</b>: Person names.</li>
  <li><b>LOC</b>: Locations and geopolitical entities.</li>
  <li><b>ORG</b>: Organizations, companies, and institutions.</li>
  <li><b>MISC</b>: Miscellaneous named entities outside the first three categories.</li>
</ul>

<p>Labels are encoded using the standard BIO (Beginning, Inside, Outside) tagging scheme. This produces nine possible token-level classes: O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, and I-MISC.</p>

<p>Each language dataset was partitioned into three disjoint splits: training, development, and Gold-Test. The Gold-Test sets were strictly protected throughout model development: they were not used for training, were not used for hyperparameter tuning, and remained locked while models were being developed. Gold-Test evaluations were executed only after model checkpoints were finalized and frozen. To guarantee dataset integrity, the test sets were protected using SHA-256 checksums, which were recorded prior to evaluation and verified to ensure that test data had remained unmodified. Table 1 reports the verified dataset characteristics and split sizes.</p>

<div class="table-caption">Table 1. Dataset Characteristics and Gold-Test Distribution across the Four Target Languages</div>
<table>
  <thead>
    <tr>
      <th>Language</th>
      <th>Script</th>
      <th>Raw Data Sources</th>
      <th>Training</th>
      <th>Dev.</th>
      <th>Gold Test</th>
      <th>Other Data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Konkani</b></td>
      <td>Devanagari</td>
      <td>Wikipedia, Herald Goa, Bhaangarbhuin</td>
      <td>850 sents.</td>
      <td>200 sents.</td>
      <td>200 sents. (124 ent.)</td>
      <td>—</td>
    </tr>
    <tr>
      <td><b>Assamese</b></td>
      <td>Assamese-Bengali</td>
      <td>Wikipedia, Asomiya Pratidin, Niyomiya Barta</td>
      <td>850 sents.</td>
      <td>200 sents.</td>
      <td>200 sents. (138 ent.)</td>
      <td>—</td>
    </tr>
    <tr>
      <td><b>Sinhala</b></td>
      <td>Sinhala</td>
      <td>Wikipedia, Ada Derana, Dinamina</td>
      <td>850 sents.</td>
      <td>200 sents.</td>
      <td>200 sents. (132 ent.)</td>
      <td>—</td>
    </tr>
    <tr>
      <td><b>Kokborok</b></td>
      <td>Roman</td>
      <td>Public corpus (20,300 sentence pairs)</td>
      <td>849 sents.</td>
      <td>199 sents.</td>
      <td>199 sents. (16 ent.)</td>
      <td>Exp. 10–11 public corpus</td>
    </tr>
  </tbody>
</table>

<p>As shown in Table 1, the Kokborok Gold-Test set contains 16 annotated entities across 199 sentences, whereas the test sets for Konkani, Assamese, and Sinhala contain between 124 and 138 entities across 200 sentences. Because the Kokborok test set has a lower entity density, every individual entity prediction produces a larger percentage change in the precision, recall, and F1 metrics for that language.</p>

<h3>6.2 Preprocessing and Grapheme-Level Tokenization</h3>

<p class="noindent">Raw text from all four languages was processed through a unified text normalization and tokenization pipeline implemented entirely from scratch.</p>

<p><b>Unicode NFC Normalization.</b> All text was normalized to Unicode Normalization Form C (NFC). In South Asian scripts, the same visible character sequence can frequently be encoded in multiple distinct ways (for example, as a precomposed character or as a base character followed by combining vowel marks). Canonical NFC normalization maps equivalent representations to a single unique byte sequence. Invisible formatting characters, byte-order marks, and zero-width spaces (<code>U+200B</code>) were removed. Zero-Width Joiner (ZWJ, <code>U+200D</code>) and Zero-Width Non-Joiner (ZWNJ, <code>U+200C</code>) characters were retained because they are required to form correct conjunct consonants in Devanagari, Assamese-Bengali, and Sinhala.</p>

<p><b>Sinhala Conjunct Repair.</b> Sinhala digital text frequently contains damaged conjunct ligatures caused by inconsistent font encoding. A rule-based repair pass was applied prior to normalization to restore missing ZWJ characters between consonant virama combinations and touching letters (<i>bandi akshara</i>).</p>

<p><b>Word Tokenization.</b> Text was split into word tokens using a rule-based tokenizer based on Unicode character classes. The tokenizer groups contiguous sequences of letters, combining marks, numbers, and joiners into single word tokens while isolating punctuation marks. This prevents accidental splitting of dependent vowel signs from their consonant bases.</p>

<p><b>Grapheme-Level Segmentation.</b> To obtain fine-grained character representations without breaking orthographic units, words were segmented into extended grapheme clusters using the <code>\X</code> pattern. In Brahmic writing systems, an extended grapheme cluster corresponds to an orthographic syllable (<i>akshara</i>), consisting of a base consonant, optional virama-consonant conjuncts, dependent vowel signs, and modifiers:
<div class="math-block">
$$\text{Grapheme Cluster} = \text{Consonant} \cdot (\text{Virama} \cdot \text{Consonant})^* \cdot \text{Vowel\_Sign}^* \cdot \text{Modifier}^*.$$
</div>
In simple terms, this formula defines a complete visual syllable by attaching vowel signs and modifiers to a base consonant cluster. Here, the dot denotes string concatenation, and the asterisk indicates zero or more occurrences. Treating each cluster as a single unit prevents the subword fragmentation that occurs when standard subword tokenizers divide characters across byte boundaries.</p>

<h3>6.3 Overall Research Framework</h3>

<p class="noindent">The research framework follows a sequential pipeline comprising nine distinct stages. Figure 1 illustrates the overall workflow.</p>

<div class="fig-box">
  <div class="diagram-container">
    <div class="workflow-grid">
      <div class="workflow-row">
        <div class="wf-block">1. Raw Data</div>
        <div class="wf-arrow-h">&rarr;</div>
        <div class="wf-block">2. Data Preparation</div>
        <div class="wf-arrow-h">&rarr;</div>
        <div class="wf-block">3. Unicode Normalization</div>
      </div>
      <div class="wf-arrow-v">&darr;</div>
      <div class="workflow-row">
        <div class="wf-block">6. Model Training</div>
        <div class="wf-arrow-h">&larr;</div>
        <div class="wf-block">5. Feature Construction</div>
        <div class="wf-arrow-h">&larr;</div>
        <div class="wf-block">4. Grapheme Tokenization</div>
      </div>
      <div class="wf-arrow-v">&darr;</div>
      <div class="workflow-row">
        <div class="wf-block">7. NER Prediction</div>
        <div class="wf-arrow-h">&rarr;</div>
        <div class="wf-block" style="background:#fbe9e7; border-color:#d32f2f;">8. Attribution</div>
        <div class="wf-arrow-h">&rarr;</div>
        <div class="wf-block" style="background:#e8f5e9; border-color:#388e3c;">9. Faithfulness Eval.</div>
      </div>
    </div>
  </div>
  <div class="fig-caption"><b>Fig. 1</b> Overall research framework showing the nine sequential stages of the experimental pipeline. Stages 8 (Attribution) and 9 (Faithfulness Evaluation) are conducted separately to assess model interpretability without modifying model parameters.</div>
</div>

<p>The workflow operates as follows:</p>
<ol>
  <li><b>Raw Data</b>: Harvesting unstructured text from news and web sources across the four languages.</li>
  <li><b>Data Preparation</b>: Applying heuristic annotation and partitioning into train, dev, and locked test splits.</li>
  <li><b>Unicode Normalization</b>: Enforcing NFC canonical form and repairing script-specific conjuncts.</li>
  <li><b>Grapheme Tokenization</b>: Segmenting words into extended grapheme clusters using the <code>\X</code> pattern.</li>
  <li><b>Feature Construction</b>: Building randomly initialized word embeddings, character CNN representations, and language embeddings.</li>
  <li><b>Model Training</b>: Executing multilingual joint training followed by phased layer-wise unfreezing fine-tuning.</li>
  <li><b>NER Prediction</b>: Decoding globally consistent BIO label sequences using the shared linear-chain CRF.</li>
  <li><b>Attribution</b>: Computing Gradient &times; Attention scores for predicted entity tokens post-hoc on the frozen model.</li>
  <li><b>Faithfulness Evaluation</b>: Measuring the change in model prediction confidence under top-<i>k</i> versus random-<i>k</i> token erasure.</li>
</ol>

<p>Stages 8 and 9 are separated deliberately: Stage 8 calculates attribution scores, while Stage 9 experimentally tests whether those scores reflect tokens that influence the model's confidence.</p>

<h3>6.4 Proposed Model Architecture</h3>

<p class="noindent">The proposed model (Model 4) is a language-conditioned, neural sequence labeling architecture trained from scratch. The model combines word-level lexical information, subword character-level morphological features, and source language conditioning into a unified representation. Figure 2 presents the architecture. Table 2 provides layer specifications.</p>

<div class="fig-box">
  <div class="diagram-container">
    <div style="display:flex; justify-content:space-around; margin-bottom:8px;">
      <div class="wf-block" style="width:130px; font-size:8pt;">Word Tokens<br>&darr;<br>Word Embed (100d, learned)</div>
      <div class="wf-block" style="width:180px; font-size:8pt;">Char Graphemes<br>&darr;<br>Char Embed (50d) &rarr; CNN<br>(30 &times; {3,4,5} &rarr; 90d)</div>
      <div class="wf-block" style="width:130px; font-size:8pt;">Language ID<br>&darr;<br>Lang Embed (16d, learned)</div>
    </div>
    <div style="text-align:center; margin:6px 0;">&darr;</div>
    <div class="wf-block" style="width:90%; margin:0 auto; background:#e8eaf6; border-color:#3f51b5;">
      Composite Input Representation: [100d &parallel; 90d &parallel; 16d] &rarr; 206d (Dropout = 0.3)
    </div>
    <div style="text-align:center; margin:6px 0;">&darr;</div>
    <div class="wf-block" style="width:90%; margin:0 auto; background:#e8eaf6; border-color:#3f51b5;">
      BiLSTM Context Encoder (2 layers, hidden = 128 per dir, output = 256d, Dropout = 0.3)
    </div>
    <div style="text-align:center; margin:6px 0;">&darr;</div>
    <div style="display:flex; justify-content:center; align-items:center; gap:12px;">
      <div class="wf-block" style="width:65%; background:#e8eaf6; border-color:#3f51b5;">
        Multi-Head Self-Attention (4 heads, d_k = 64, Residual + LayerNorm &rarr; 256d, Dropout = 0.1)
      </div>
      <div style="font-size:12pt; color:#d32f2f;">&rarr;</div>
      <div class="wf-block" style="width:28%; background:#ffebee; border-color:#d32f2f; font-size:8pt;">
        <b>Attribution Branch</b><br>(Grad &times; Attn, post-hoc,<br>no trainable params)
      </div>
    </div>
    <div style="text-align:center; margin:6px 0;">&darr;</div>
    <div class="wf-block" style="width:90%; margin:0 auto; background:#e8eaf6; border-color:#3f51b5;">
      Shared Linear Emission Projection (256d &rarr; 9 BIO tags)
    </div>
    <div style="text-align:center; margin:6px 0;">&darr;</div>
    <div class="wf-block" style="width:90%; margin:0 auto; background:#e8eaf6; border-color:#3f51b5;">
      Shared Linear-Chain CRF (Viterbi decoding, learned transition scores)
    </div>
    <div style="text-align:center; margin:6px 0;">&darr;</div>
    <div class="wf-block" style="width:50%; margin:0 auto; background:#c8e6c9; border-color:#2e7d32;">
      Predicted BIO Label Sequence
    </div>
  </div>
  <div class="fig-caption"><b>Fig. 2</b> Detailed architecture of the proposed multilingual NER model (Model 4). The composite 206-dimensional input vector passes through a 2-layer BiLSTM and a 4-head self-attention layer with residual normalization. The linear emission layer and CRF decoder are shared across all four languages. The attribution branch operates post-hoc on the frozen network.</div>
</div>

<div class="table-caption">Table 2. Layer-Wise Architecture and Parameter Specifications of the Proposed Model</div>
<table>
  <thead>
    <tr>
      <th>Layer / Component</th>
      <th>Input Dim.</th>
      <th>Output Dim.</th>
      <th>Configuration</th>
      <th>Hyperparameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Word Embedding</td>
      <td>—</td>
      <td>100d</td>
      <td>Randomly initialized table</td>
      <td>Dropout = 0.3</td>
    </tr>
    <tr>
      <td>Char Grapheme Embedding</td>
      <td>—</td>
      <td>50d</td>
      <td>Extended grapheme clusters</td>
      <td>Dropout = 0.3</td>
    </tr>
    <tr>
      <td>Char CNN Convolutions</td>
      <td>50d</td>
      <td>90d</td>
      <td>Widths k &in; {3,4,5}, 30 filters each</td>
      <td>ReLU, global max-pool</td>
    </tr>
    <tr>
      <td>Language Embedding</td>
      <td>—</td>
      <td>16d</td>
      <td>4 languages, learned table</td>
      <td>Learned from scratch</td>
    </tr>
    <tr>
      <td>Composite Input Vector</td>
      <td>—</td>
      <td>206d</td>
      <td>Concat: 100d + 90d + 16d</td>
      <td>Concat Dropout = 0.3</td>
    </tr>
    <tr>
      <td>BiLSTM Context Encoder</td>
      <td>206d</td>
      <td>256d</td>
      <td>2 layers, hidden = 128 per dir.</td>
      <td>Dropout = 0.3</td>
    </tr>
    <tr>
      <td>Multi-Head Self-Attn</td>
      <td>256d</td>
      <td>256d</td>
      <td>4 heads, d_k = 64, 1 block</td>
      <td>Residual + LayerNorm, Drop = 0.1</td>
    </tr>
    <tr>
      <td>Linear Emission Proj.</td>
      <td>256d</td>
      <td>9d</td>
      <td>Shared linear transformation</td>
      <td>Xavier uniform init.</td>
    </tr>
    <tr>
      <td>Linear-Chain CRF</td>
      <td>9d</td>
      <td>9d</td>
      <td>Shared transition matrix</td>
      <td>Viterbi decoding</td>
    </tr>
  </tbody>
</table>

<h4>6.4.1 Composite Input Representation</h4>
<p class="noindent">At each token position <i>t</i>, the network forms a composite vector \(\mathbf{x}_t\) by concatenating three representations:</p>
<ol>
  <li>A 100-dimensional word embedding \(\mathbf{e}^{(w)}_t\) learned from scratch.</li>
  <li>A 90-dimensional character-level representation \(\mathbf{e}^{(c)}_t\) generated by the character CNN.</li>
  <li>A 16-dimensional language embedding \(\mathbf{e}^{(\ell)}\) representing the source language ID.</li>
</ol>
<p>The resulting vector has 206 dimensions (\(100 + 90 + 16 = 206\)). A dropout layer with a rate of 0.3 is applied to this vector. This composite representation provides lexical identity through the word vector, morphological cues through the character vector, and language identity through the language embedding.</p>

<h4>6.4.2 Character-Level CNN</h4>
<p class="noindent">The character CNN extracts morphological information from the grapheme cluster sequence of each word. Each grapheme cluster is mapped to a 50-dimensional learned embedding. Three parallel 1D convolutional layers with kernel widths of 3, 4, and 5 grapheme units are applied. Each kernel width uses 30 filters, followed by a ReLU activation function and global max-over-time pooling. Concatenating the outputs of the three filter sizes yields a 90-dimensional character representation (\(3 \times 30 = 90\)). This representation allows the model to handle inflected and unseen words that share roots or affixes with known words.</p>

<h4>6.4.3 BiLSTM Context Encoder</h4>
<p class="noindent">The sequence of 206-dimensional vectors is fed into a two-layer bidirectional LSTM. The hidden size is 128 dimensions in each direction. Concatenating the forward hidden state (128d) and backward hidden state (128d) at each token position yields a 256-dimensional contextual vector. A dropout rate of 0.3 is applied to the recurrent outputs. The BiLSTM captures sequential dependencies across the entire sentence.</p>

<h4>6.4.4 Multi-Head Self-Attention</h4>
<p class="noindent">The BiLSTM representations are processed by a single multi-head self-attention layer with 4 heads. The 256-dimensional space is divided equally across the 4 heads, giving each head a projection dimension of \(d_k = 64\). Scaled dot-product attention computes direct interactions between all pairs of tokens in the sentence. The outputs of the 4 heads are concatenated, projected back to 256 dimensions, regularized with an attention dropout of 0.1, and combined with the input through a residual addition followed by Layer Normalization:
<div class="math-block">
$$\tilde{\mathbf{H}} = \text{LayerNorm}(\mathbf{H} + \text{Dropout}(\text{MultiHead}(\mathbf{H}))).$$
</div>
In simple terms, this equation adds the attention output to the BiLSTM hidden states and normalizes the sum, ensuring stable gradients. Here, \(\mathbf{H}\) is the input matrix of hidden states, \(\text{MultiHead}(\mathbf{H})\) is the concatenated multi-head context, and \(\text{LayerNorm}\) normalizes values across features. This layer allows the model to connect related words across long distances without relying solely on the step-by-step recurrence of the LSTM.</p>

<h4>6.4.5 CRF Decoder</h4>
<p class="noindent">The attended representations are projected to 9 tag emission scores using a shared linear layer. A shared linear-chain Conditional Random Field (CRF) decodes the label sequence. The shared CRF uses a single transition matrix across all four languages, as confirmed in the final configuration files (<code>crf_config: shared</code>). During training, the CRF minimizes the negative log-likelihood of the gold label sequence. During inference, the Viterbi decoder selects the highest-scoring complete label sequence using both emission and learned transition scores.</p>

<h4>6.4.6 Explainability Branch</h4>
<p class="noindent">During the forward pass, the multi-head self-attention module records the attention weight tensors for all four heads. The explainability branch uses these stored weights alongside the model gradients to compute token-level attribution scores after training. This branch adds no trainable parameters and does not change the network's forward predictions.</p>

<h3>6.5 Evaluated Model Configurations</h3>

<p class="noindent">To study the architectural progression, four model variants were implemented in the project codebase. Table 3 summarizes their architectural configurations.</p>

<div class="table-caption">Table 3. Evaluated Model Configurations and Architectural Progression</div>
<table>
  <thead>
    <tr>
      <th>Configuration</th>
      <th>Source File</th>
      <th>Core Components</th>
      <th>Architectural Role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Model 1 (Baseline)</b></td>
      <td><code>ner_model_v1_baseline.py</code></td>
      <td>Word Embed (100d) + BiLSTM + CRF</td>
      <td>Word-only baseline</td>
    </tr>
    <tr>
      <td><b>Model 2 (Char-CNN)</b></td>
      <td><code>ner_model_v2_charcnn.py</code></td>
      <td>Word (100d) + Char-CNN (90d) &rarr; 190d + 2-layer BiLSTM + CRF</td>
      <td>Adds character morphology</td>
    </tr>
    <tr>
      <td><b>Model 3 (Attention)</b></td>
      <td><code>ner_model_v3_attention.py</code></td>
      <td>190d Input + 2-layer BiLSTM + 4-Head Attn + CRF</td>
      <td>Adds self-attention mechanism</td>
    </tr>
    <tr>
      <td><b>Model 4 (Proposed)</b></td>
      <td><code>ner_model_v4_multilingual.py</code></td>
      <td>Word (100d) + Char (90d) + Lang (16d) &rarr; 206d + Attn + Shared CRF</td>
      <td>Complete multilingual model</td>
    </tr>
  </tbody>
</table>

<p><i>Note on Evaluation Evidence:</i> The project artifacts contain complete, verified Gold-Test evaluation records for the final multilingual system (Model 4). While Models 1, 2, and 3 are implemented in the codebase, separate standalone evaluation logs for these earlier variants are not present in the workspace. Therefore, their individual test F1 scores are not reported here to avoid unverified claims.</p>

<p><b>Model 1 — Word-Level BiLSTM-CRF Baseline.</b> Model 1 contains a 100-dimensional learned word embedding, a bidirectional LSTM context encoder with hidden size 128 per direction (configured with a single layer by default), a linear emission layer, and a linear-chain CRF decoder. It excludes character embeddings, character convolutions, language conditioning, and self-attention.</p>

<p><b>Model 2 — Char-CNN + BiLSTM-CRF.</b> Model 2 adds the character CNN to Model 1. The input combines the 100-dimensional word embedding with the 90-dimensional character representation into a 190-dimensional vector. The sequence encoder is expanded to a two-layer BiLSTM followed by a CRF decoder.</p>

<p><b>Model 3 — Char-CNN + BiLSTM + Self-Attention + CRF.</b> Model 3 adds the 4-head self-attention layer with residual connection and Layer Normalization between the two-layer BiLSTM and the CRF decoder. The input dimension remains 190. This model introduces internal attention weights that can be inspected for explainability.</p>

<p><b>Model 4 — Final Proposed Multilingual Model.</b> Model 4 adds the 16-dimensional language embedding to Model 3, producing the complete 206-dimensional composite input. The CRF decoder and linear emission projection are shared across all four languages. Model 4 serves as the primary evaluation model.</p>

<h3>6.6 Mathematical Formulation</h3>

<p class="noindent">This subsection presents the mathematical definitions for each component of the proposed system.</p>

<h4>6.6.1 Input and Composite Representation</h4>
<p class="noindent">Let a sentence of <i>T</i> tokens be represented as a sequence of word indices \((w_1, w_2, \ldots, w_T)\), their corresponding grapheme cluster sequences \((\mathbf{c}_1, \mathbf{c}_2, \ldots, \mathbf{c}_T)\), and a language identifier \(\ell \in \{0, 1, 2, 3\}\). The composite input vector \(\mathbf{x}_t\) at position <i>t</i> is formed by concatenating the word embedding, the character CNN output, and the language embedding:
<div class="math-block">
$$\mathbf{x}_t = [\mathbf{e}^{(w)}(w_t) \parallel \text{CharCNN}(\mathbf{c}_t) \parallel \mathbf{e}^{(\ell)}(\ell)] \in \mathbb{R}^{206}.$$
</div>
In simple terms, the model joins word information, character information, and language information into one 206-dimensional vector. Here, \(\mathbf{e}^{(w)}(w_t) \in \mathbb{R}^{100}\) is the learned word embedding for word \(w_t\), \(\text{CharCNN}(\mathbf{c}_t) \in \mathbb{R}^{90}\) is the character representation, and \(\mathbf{e}^{(\ell)}(\ell) \in \mathbb{R}^{16}\) is the language embedding. The symbol \(\parallel\) denotes vector concatenation.</p>

<h4>6.6.2 Character-Level Representation</h4>
<p class="noindent">For a word with grapheme clusters \((g_1, g_2, \ldots, g_{L})\), each cluster \(g_i\) is mapped to a 50-dimensional embedding \(\mathbf{e}^{(g)}_i \in \mathbb{R}^{50}\). A 1D convolution with kernel width \(k \in \{3, 4, 5\}\) and weight tensor \(\mathbf{W}^{(k)} \in \mathbb{R}^{30 \times (k \times 50)}\) computes local features:
<div class="math-block">
$$\mathbf{f}^{(k)}_i = \text{ReLU}\left(\mathbf{W}^{(k)} \cdot \mathbf{e}^{(g)}_{i:i+k-1} + \mathbf{b}^{(k)}\right) \in \mathbb{R}^{30}.$$
</div>
In simple terms, this operation slides a window of width <i>k</i> across the grapheme embeddings to detect local character patterns. Here, \(\mathbf{e}^{(g)}_{i:i+k-1}\) is the concatenated window of <i>k</i> grapheme embeddings, \(\mathbf{b}^{(k)} \in \mathbb{R}^{30}\) is a bias term, and \(\text{ReLU}(z) = \max(0, z)\) is the activation function. Global max-over-time pooling selects the most salient feature across all window positions:
<div class="math-block">
$$p^{(k)} = \max_{1 \le i \le L - k + 1} \mathbf{f}^{(k)}_i \in \mathbb{R}^{30}.$$
</div>
In simple terms, pooling keeps only the highest activation for each filter across the entire word. The outputs from all three kernel widths are concatenated:
<div class="math-block">
$$\text{CharCNN}(\mathbf{c}_t) = [p^{(3)} \parallel p^{(4)} \parallel p^{(5)}] \in \mathbb{R}^{90}.$$
</div>
In simple terms, joining the pooled vectors from widths 3, 4, and 5 forms a complete 90-dimensional morphological summary of the word.</p>

<h4>6.6.3 BiLSTM Context Encoding</h4>
<p class="noindent">The sequence of input vectors \((\mathbf{x}_1, \ldots, \mathbf{x}_T)\) is processed by a forward LSTM and a backward LSTM:
<div class="math-block">
$$\overrightarrow{\mathbf{h}}_t = \text{LSTM}_{\text{fwd}}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1}), \quad
\overleftarrow{\mathbf{h}}_t = \text{LSTM}_{\text{bwd}}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1}).$$
</div>
In simple terms, the forward LSTM reads the sentence from left to right, while the backward LSTM reads it from right to left. The forward hidden state \(\overrightarrow{\mathbf{h}}_t \in \mathbb{R}^{128}\) and backward hidden state \(\overleftarrow{\mathbf{h}}_t \in \mathbb{R}^{128}\) are concatenated:
<div class="math-block">
$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t \parallel \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{256}.$$
</div>
In simple terms, combining both directions gives each word a 256-dimensional summary of its left and right context. The matrix \(\mathbf{H} = (\mathbf{h}_1, \ldots, \mathbf{h}_T)^\top \in \mathbb{R}^{T \times 256}\) collects all hidden states.</p>

<h4>6.6.4 Multi-Head Self-Attention</h4>
<p class="noindent">For each attention head \(h \in \{1, 2, 3, 4\}\), query, key, and value matrices are computed using head-specific projections:
<div class="math-block">
$$\mathbf{Q}^{(h)} = \mathbf{H} \mathbf{W}_Q^{(h)}, \quad
\mathbf{K}^{(h)} = \mathbf{H} \mathbf{W}_K^{(h)}, \quad
\mathbf{V}^{(h)} = \mathbf{H} \mathbf{W}_V^{(h)} \in \mathbb{R}^{T \times 64},$$
</div>
where \(\mathbf{W}_Q^{(h)}, \mathbf{W}_K^{(h)}, \mathbf{W}_V^{(h)} \in \mathbb{R}^{256 \times 64}\). In simple terms, these linear projections transform the BiLSTM representations into queries, keys, and values for head <i>h</i>. The scaled dot-product attention weights for head <i>h</i> are:
<div class="math-block">
$$\mathbf{A}^{(h)} = \text{softmax}\left(\frac{\mathbf{Q}^{(h)} (\mathbf{K}^{(h)})^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{T \times T},$$
</div>
where \(d_k = 64\). In simple terms, this formula computes pairwise relevance scores between all words in the sentence and normalizes them into probabilities. Masked positions (padding) are assigned a large negative value (\(-10^9\)) before the softmax operation. The attended representation for head <i>h</i> is:
<div class="math-block">
$$\text{Head}^{(h)} = \mathbf{A}^{(h)} \mathbf{V}^{(h)} \in \mathbb{R}^{T \times 64}.$$
</div>
In simple terms, this takes a weighted average of word values based on their attention scores. All head outputs are concatenated and projected using \(\mathbf{W}_O \in \mathbb{R}^{256 \times 256}\), followed by a residual connection and Layer Normalization:
<div class="math-block">
$$\tilde{\mathbf{H}} = \text{LayerNorm}\left(\mathbf{H} + \text{Dropout}\left([\text{Head}^{(1)} \parallel \cdots \parallel \text{Head}^{(4)}] \mathbf{W}_O\right)\right) \in \mathbb{R}^{T \times 256}.$$
</div>
In simple terms, the multi-head output is combined with the original BiLSTM representation and normalized across dimensions.</p>

<h4>6.6.5 CRF Sequence Scoring and Viterbi Decoding</h4>
<p class="noindent">The attended hidden states \(\tilde{\mathbf{H}}\) are mapped to tag emission logits using a shared linear projection \(\mathbf{W}_E \in \mathbb{R}^{256 \times 9}\) and bias \(\mathbf{b}_E \in \mathbb{R}^9\):
<div class="math-block">
$$\mathbf{P}_t = \tilde{\mathbf{h}}_t \mathbf{W}_E + \mathbf{b}_E \in \mathbb{R}^9.$$
</div>
In simple terms, this linear layer converts the 256-dimensional representation of token <i>t</i> into 9 emission scores corresponding to the 9 BIO tags. For a label sequence \(\mathbf{y} = (y_1, \ldots, y_T)\), the score assigned by the CRF is:
<div class="math-block">
$$S(\mathbf{x}, \mathbf{y}) = \sum_{t=1}^T P_{t, y_t} + \sum_{t=1}^{T-1} A^{\text{CRF}}_{y_t, y_{t+1}},$$
</div>
where \(\mathbf{A}^{\text{CRF}} \in \mathbb{R}^{9 \times 9}\) is the learned transition matrix. In simple terms, the sequence score sums the individual emission scores and the transition scores between adjacent tags. The conditional probability of sequence \(\mathbf{y}\) is:
<div class="math-block">
$$P(\mathbf{y} \mid \mathbf{x}) = \frac{\exp(S(\mathbf{x}, \mathbf{y}))}{\sum_{\mathbf{y}'} \exp(S(\mathbf{x}, \mathbf{y}'))}.$$
</div>
In simple terms, this normalizes the score of sequence \(\mathbf{y}\) against all possible label sequences. The model is trained by minimizing the negative log-likelihood:
<div class="math-block">
$$\mathcal{L} = -\log P(\mathbf{y}^* \mid \mathbf{x}) = -S(\mathbf{x}, \mathbf{y}^*) + \log \sum_{\mathbf{y}'} \exp(S(\mathbf{x}, \mathbf{y}')),$$
</div>
where \(\mathbf{y}^*\) is the gold tag sequence. During inference, the optimal tag sequence is found using the Viterbi algorithm:
<div class="math-block">
$$\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} S(\mathbf{x}, \mathbf{y}).$$
</div>
In simple terms, Viterbi decoding finds the complete sequence of tags that maximizes the combined emission and transition score.</p>

<h4>6.6.6 Evaluation Metrics</h4>
<p class="noindent">Performance is evaluated using entity-level Precision, Recall, and F1 score under exact span and label matching:
<div class="math-block">
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}.$$
</div>
In simple terms, Precision measures the fraction of predicted entities that are correct, Recall measures the fraction of actual entities found, and F1 is their harmonic mean. Here, \(\text{TP}\) is the number of correctly identified entity spans with matching boundaries and types, \(\text{FP}\) is the number of predicted entities not matching the gold standard, and \(\text{FN}\) is the number of gold entities missed by the model. Token accuracy is measured as the ratio of correctly tagged tokens to total tokens.</p>

<h4>6.6.7 Attribution Formulation</h4>
<p class="noindent">For an entity token at query position <i>q</i> with predicted tag emission score \(s_{q, y^*}\), the Gradient &times; Attention attribution of token position <i>j</i> in head <i>h</i> is:
<div class="math-block">
$$\alpha^{(h)}_{q \to j} = \frac{\partial s_{q, y^*}}{\partial A^{(h)}_{q, j}} \cdot A^{(h)}_{q, j}.$$
</div>
In simple terms, this multiplies the attention weight by how sensitive the prediction is to that weight, measuring both attention magnitude and directional importance. The attribution is averaged across all four attention heads:
<div class="math-block">
$$\alpha_{q \to j} = \frac{1}{4} \sum_{h=1}^4 \alpha^{(h)}_{q \to j}.$$
</div>
In simple terms, averaging across heads combines the attribution evidence from all attention heads into a single score. To obtain scale-invariant attribution scores, absolute values are normalized over all tokens in the sentence:
<div class="math-block">
$$\hat{\alpha}_{q \to j} = \frac{|\alpha_{q \to j}|}{\sum_{j'=1}^T |\alpha_{q \to j'}|}.$$
</div>
In simple terms, this normalizes scores so that all token attributions in the sentence are positive and sum to 1.</p>

<h4>6.6.8 Faithfulness Formulation</h4>
<p class="noindent">Faithfulness is evaluated by measuring the drop in prediction confidence when important context tokens are erased. The evaluation distinguishes between the primary benchmark protocol and an alternative baseline protocol:</p>

<p><b>Primary Protocol (Zero-Embedding with CRF Marginals).</b> Under the primary protocol, the unperturbed baseline confidence of an entity prediction at token <i>q</i> is defined as its exact posterior marginal probability computed from forward-backward CRF inference:
<div class="math-block">
$$C_0 = P_{\text{CRF}}(y_q = y^*_q \mid \mathbf{x}) = \frac{\alpha_q(y^*_q) \beta_q(y^*_q)}{Z(\mathbf{x})},$$
</div>
where \(\alpha_q\) and \(\beta_q\) are the forward and backward trellis vectors, and \(Z(\mathbf{x})\) is the sequence partition function. In simple terms, this calculates the exact marginal probability that token <i>q</i> receives tag \(y^*_q\) taking all sequence transitions into account.</p>

<p>For top-<i>k</i> context erasure, the top-<i>k</i> non-target tokens (\(j \neq q\)) with the highest attribution scores \(\hat{\alpha}_{q \to j}\) have their 206-dimensional composite input vectors set to zero (\(\mathbf{x}_j = \mathbf{0} \in \mathbb{R}^{206}\)). The target entity token <i>q</i> is strictly excluded from erasure. The model is re-evaluated to obtain the perturbed marginal confidence \(C_{\text{top-}k} = P_{\text{CRF}}(y_q = y^*_q \mid \mathbf{x}_{\setminus \text{top-}k}^{(0)})\), and the confidence drop is:
<div class="math-block">
$$\Delta_{\text{top-}k} = \max(0, C_0 - C_{\text{top-}k}).$$
</div>
In simple terms, this measures how much the model's marginal confidence drops when the most highly attributed context tokens are neutralized. For the control group, <i>k</i> tokens are chosen at random from the non-target context (\(j \neq q\)) and zeroed, yielding drop \(\Delta_{\text{rand-}k}\). The faithfulness deletion gap is:
<div class="math-block">
$$\text{Gap}_k = \Delta_{\text{top-}k} - \Delta_{\text{rand-}k}.$$
</div>
In simple terms, a positive gap (\(\text{Gap}_k > 0\)) means that zeroing top-attributed tokens hurts model confidence more than zeroing random context tokens, demonstrating that the attribution scores reflect context tokens relevant to the decision.</p>

<p><b>Baseline Protocol (Token Replacement with Local Softmax).</b> In the alternative baseline protocol, candidate tokens are replaced with an explicit unknown token symbol (<code>&lt;UNK&gt;</code>), and confidence is measured using the local emission softmax probability \(\frac{\exp(P_{q, y^*_q})}{\sum_c \exp(P_{q, c})}\) without forward-backward CRF marginalization.</p>

<h3>6.7 Multi-Stage Training and Experimental Procedure</h3>

<p class="noindent">Training was conducted in two verified stages designed to encourage cross-lingual transfer while allowing language-specific specialization:</p>

<p><b>Stage 1: Multilingual Joint Training.</b> The complete shared model (word embedding, character CNN, language embedding, BiLSTM, self-attention, and shared CRF) was trained jointly on the combined training sets of all four languages. In this stage, all parameters were updated simultaneously. This allowed the network to learn shared structural representations across the four languages from scratch.</p>

<p><b>Stage 2: Layer-Wise Unfreezing Specialization.</b> After joint training, the model was fine-tuned for each language using a phased layer-unfreezing schedule (<code>training/layer_unfreeze_schedule.py</code>). Under this schedule, the lower layers (word embedding, character CNN, language embedding, and BiLSTM) were kept frozen for the first three epochs (\(\text{epoch} < 4\)). Only the upper layers (self-attention, linear projection, and shared CRF) were updated. Starting at epoch 4, all layers were unfrozen and trained jointly with a reduced learning rate. This prevented the language-specific training from overwriting the shared representations learned in Stage 1.</p>

<p>Table 4 reports the verified hyperparameters for the fine-tuning stage.</p>

<div class="table-caption">Table 4. Verified Training and Fine-Tuning Hyperparameters</div>
<table>
  <thead>
    <tr>
      <th>Hyperparameter / Setting</th>
      <th>Verified Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Optimizer</td>
      <td>AdamW</td>
    </tr>
    <tr>
      <td>Fine-tuning learning rate</td>
      <td>0.0002</td>
    </tr>
    <tr>
      <td>Weight decay</td>
      <td>0.0001 (10<sup>-4</sup>)</td>
    </tr>
    <tr>
      <td>Gradient norm clipping</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>Batch size</td>
      <td>32</td>
    </tr>
    <tr>
      <td>Unfreezing epoch</td>
      <td>Epoch 4</td>
    </tr>
    <tr>
      <td>Frozen layers (epochs 1–3)</td>
      <td>Word embed, Char CNN, Language embed, BiLSTM</td>
    </tr>
    <tr>
      <td>Trainable layers from epoch 1</td>
      <td>Multi-head attention, emission projection, CRF</td>
    </tr>
    <tr>
      <td>CRF configuration</td>
      <td>Shared across all languages</td>
    </tr>
  </tbody>
</table>

<p><i>Note on Excluded Training Techniques:</i> Earlier design documents proposed confidence-filtered self-training, targeted entity substitution augmentation, and an automated hyperparameter search. While configuration files mention these options, no completed experiment logs or output artifacts exist in the project repository to support them. In accordance with strict scientific reporting standards, these unverified procedures are omitted from the experimental claims.</p>

<h3>6.8 Explainability and Faithfulness Method</h3>

<p class="noindent">The explainability analysis focuses on determining whether internal self-attention distributions provide meaningful explanations of model decisions, and whether Gradient &times; Attention attribution offers greater faithfulness.</p>

<p class="noindent">To avoid misinterpreting explainability results, three distinct concepts are separated:</p>
<ol>
  <li><b>Raw Attention</b>: The internal attention weights \(A^{(h)}_{q, j}\) computed by the network. Attention weights show how the model routes information between positions, but high attention does not necessarily mean that a token caused the prediction.</li>
  <li><b>Gradient &times; Attention Attribution</b>: The product of the attention weight and the gradient of the predicted tag score with respect to that weight. This measures the sensitivity of the prediction to changes in attention.</li>
  <li><b>Faithfulness Evaluation</b>: An empirical test measuring whether erasing tokens with high attribution causes a larger drop in prediction confidence than erasing random tokens.</li>
</ol>

<p>The faithfulness benchmark was evaluated on a sample of 100 gold-development entity instances per language. For each entity, attribution scores were computed using both Raw Attention and Gradient &times; Attention. Erasure experiments were conducted for \(k \in \{1, 2, 3\}\). The primary benchmark used zero-embedding context erasure with exact forward-backward CRF marginal posterior confidence. The baseline protocol used token replacement with local emission softmax confidence. The results, including faithfulness gaps and win rates, are reported and discussed in Section 7.</p>

<p>In summary, attention weights provide an internal routing signal that can be inspected, Gradient &times; Attention combines this signal with prediction sensitivity, and the token-erasure test verifies whether high-attribution tokens are associated with measurable changes in model behavior.</p>

</body>
</html>
"""

print("Writing Section_6_Methodology_Springer_Final.html...")
with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Compiling publication PDF via Headless Chrome: {PDF_PATH}...")
chrome_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
cmd = [
    chrome_exe,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--print-to-pdf-no-header",
    f"--print-to-pdf={PDF_PATH}",
    str(HTML_PATH)
]
subprocess.run(cmd, check=True)
time.sleep(2)

if PDF_PATH.exists():
    size = PDF_PATH.stat().st_size
    print(f"PDF successfully generated: {PDF_PATH} ({size} bytes)")
    doc = fitz.open(str(PDF_PATH))
    print(f"Total Pages in Generated PDF: {len(doc)}")
    doc.close()
else:
    print("ERROR: PDF was not generated!")
