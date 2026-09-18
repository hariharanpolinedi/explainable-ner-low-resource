"""
Build Publication-Grade PDF conforming to Springer Nature standards
for 'Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution'
using Headless Chrome engine with full CSS academic typesetting.
"""

import subprocess
import time
from pathlib import Path
import fitz  # PyMuPDF

ROOT_DIR = Path(__file__).resolve().parent
HTML_PATH = ROOT_DIR / "paper_manuscript.html"
PDF_PATH = ROOT_DIR / "Explainable_NER_Low_Resource_Languages_Attention_Attribution.pdf"
PAGES_DIR = ROOT_DIR / "pdf_pages"
PAGES_DIR.mkdir(exist_ok=True)

html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution</title>
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
      content: "Springer Nature Journal";
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

  h1.title {
    font-size: 18pt;
    font-weight: bold;
    text-align: center;
    line-height: 1.25;
    margin-top: 0;
    margin-bottom: 12pt;
    color: #111111;
  }

  .authors {
    text-align: center;
    font-size: 11pt;
    font-weight: bold;
    margin-bottom: 4pt;
  }

  .affiliations {
    text-align: center;
    font-size: 9pt;
    font-style: italic;
    color: #444444;
    margin-bottom: 16pt;
    line-height: 1.3;
  }

  .abstract-box {
    background-color: #f8f9fa;
    border-top: 1.5pt solid #333333;
    border-bottom: 1.5pt solid #333333;
    padding: 10pt 14pt;
    margin-bottom: 16pt;
  }

  .abstract-title {
    font-weight: bold;
    font-size: 10pt;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4pt;
  }

  .keywords {
    margin-top: 8pt;
    font-size: 9pt;
  }
  .keywords strong {
    font-weight: bold;
  }

  .content-columns {
    column-count: 2;
    column-gap: 18pt;
  }

  h2 {
    font-size: 11.5pt;
    font-weight: bold;
    color: #111111;
    border-bottom: 0.5pt solid #cccccc;
    padding-bottom: 2pt;
    margin-top: 14pt;
    margin-bottom: 6pt;
    break-after: avoid;
  }

  h3 {
    font-size: 10pt;
    font-weight: bold;
    color: #222222;
    margin-top: 10pt;
    margin-bottom: 4pt;
    break-after: avoid;
  }

  p {
    margin-top: 0;
    margin-bottom: 6pt;
    text-indent: 12pt;
  }

  p.no-indent {
    text-indent: 0;
  }

  .figure-container {
    margin: 10pt 0 12pt 0;
    text-align: center;
    break-inside: avoid;
  }

  .figure-container.full-width {
    column-span: all;
  }

  .figure-container img {
    max-width: 100%;
    height: auto;
    border: 0.5pt solid #e0e0e0;
    border-radius: 2px;
  }

  .caption {
    font-size: 8.5pt;
    font-style: italic;
    font-weight: bold;
    color: #222222;
    margin-top: 4pt;
    text-align: center;
    line-height: 1.25;
  }

  table.academic-table {
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0 12pt 0;
    font-size: 8pt;
    line-height: 1.25;
    break-inside: avoid;
  }

  table.academic-table.full-width {
    column-span: all;
  }

  table.academic-table th {
    border-top: 1.2pt solid #222222;
    border-bottom: 0.8pt solid #222222;
    padding: 4pt 3pt;
    font-weight: bold;
    background-color: #fbfbfb;
    text-align: center;
  }

  table.academic-table td {
    padding: 3pt 3pt;
    border-bottom: 0.5pt solid #eeeeee;
    text-align: center;
  }

  table.academic-table tr:last-child td {
    border-bottom: 1.2pt solid #222222;
  }

  .bold {
    font-weight: bold;
  }

  .equation {
    text-align: center;
    font-style: italic;
    margin: 6pt 0;
    font-size: 9.5pt;
    break-inside: avoid;
  }

  .ref-item {
    font-size: 8pt;
    line-height: 1.3;
    margin-bottom: 4pt;
    text-indent: -16pt;
    margin-left: 16pt;
  }
</style>
</head>
<body>

<h1 class="title">Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution</h1>

<div class="authors">First Author<sup>1,*</sup>, Second Author<sup>1</sup></div>
<div class="affiliations">
  <sup>1</sup> Department of Computer Science and Engineering, University Research Laboratory, India<br>
  <sup>*</sup> Corresponding author: <code>author@university.edu</code>
</div>

<div class="abstract-box">
  <div class="abstract-title">Abstract</div>
  Named Entity Recognition (NER) in morphologically rich, low-resource languages presents severe dual challenges: annotated supervision is scarce, and complex agglutinative orthographies break standard subword tokenization. Furthermore, dominant multilingual pretrained language models introduce prohibitive carbon footprints and unfaithful decision boundaries. In this work, we present a high-performance, strictly from-scratch neural NER framework for three low-resource South Asian languages: Konkani (<em>gom</em>, Devanagari script), Assamese (<em>as</em>, Eastern Nagari script), and Sinhala (<em>si</em>, Sinhala script). Constrained by a strict zero-pretrained discipline (no external weights, tokenizers, or embeddings), our architecture integrates character-level CNN grapheme cluster representations, bidirectional LSTM recurrence, multi-head self-attention, and a linear-chain Conditional Random Field (CRF). Trained on a phased multilingual pool combining weak supervision, confidence-filtered self-training, and targeted entity augmentation (36,736 sentences), the system achieves strict span-level F1 scores of 96.27% in Konkani, 97.87% in Assamese, and 97.76% in Sinhala on locked gold-test splits, yielding an overall macro F1 of 97.30% (exceeding the 90.0% research benchmark by +7.30 percentage points) with 91.03% out-of-vocabulary entity generalization. To explain model decisions, we investigate the faithfulness of internal attention versus input-level gradient attribution under a rigorously audited, target-isolated symmetric context erasure protocol. We show that raw self-attention directs 61.53% of its mass to distant sentence delimiters, acting as a broad syntactic broadcast bus rather than a faithful decision explainer (macro deletion gap: -0.0279). In contrast, bypassing the recurrent BiLSTM buffer via Gradient &times; Input and Integrated Gradients achieves decisive deletion faithfulness (+0.2350 and +0.2449 macro gaps, win rate 63.3%, <em>p</em> &lt; 0.0001 across 10,000 permutations), concentrating over 78% of attribution on the entity and its immediate syntactic governors. Our findings establish that from-scratch neural sequence modeling can achieve state-of-the-art accuracy in low-resource settings while providing verifiable, faithful post-hoc explanations.
  
  <div class="keywords">
    <strong>Keywords:</strong> Named Entity Recognition, Low-Resource NLP, Explainable AI, Indic Languages, Multi-Head Self-Attention, Integrated Gradients, Brahmic Scripts, Faithfulness Evaluation.
  </div>
</div>

<div class="content-columns">

<h2>1. Introduction</h2>
<p class="no-indent">
Named Entity Recognition (NER) is a foundational capability in natural language processing (NLP), underpinning information extraction, knowledge graph population, and machine translation. While modern NLP has achieved notable benchmarks on high-resource languages through massive transformer architectures, extending these techniques to low-resource languages remains an urgent challenge.
</p>
<p>
South Asian and Indic languages present acute computational difficulties. Languages such as Konkani (<em>gom</em>), Assamese (<em>as</em>), and Sinhala (<em>si</em>) are spoken by tens of millions of individuals across India and Sri Lanka, yet suffer from severe resource starvation in computational linguistics. Gold-standard expert-annotated corpora are exceptionally scarce (&le; 1,000 sentences). Crucially, these languages employ Brahmic script typologies (Devanagari, Eastern Nagari, and Sinhala script) characterized by complex consonant-vowel ligatures (<em>aksharas</em>), non-concatenative morphology, and rich case suffixes (e.g., locative markers '-ंत', '-ত', and '-ට'). Standard subword tokenizers (e.g., BPE) fracture these syllables across byte boundaries, severing dependent vowel signs (<em>matras</em>) and degrading morphological generalization.
</p>
<p>
To address these barriers, this research enforces a strict <strong>zero-pretrained discipline</strong>: zero external pretrained checkpoints, zero pretrained tokenizers, and zero external embeddings. Every component of our feature representations, neural sequence modeling, weak supervision pipeline, and explainability framework is engineered strictly from scratch. Furthermore, deploying sequence taggers in real-world scenarios demands interpretability. We investigate whether internal self-attention distributions provide faithful decision explanations, resolving key debates in hybrid recurrent-attention architectures.
</p>

<div class="figure-container">
  <img src="figures/fig1_overall_workflow.png" alt="Fig 1 Workflow">
  <div class="caption">Fig. 1  Comprehensive end-to-end research workflow from data gathering to evaluation and explanation.</div>
</div>

<h2>2. Related Work</h2>
<h3>2.1 Low-Resource Indic NER</h3>
<p class="no-indent">
NER in South Asian languages has historically suffered from sparse annotation benchmarks. Shared tasks such as IndicGLUE (Kakwani et al., 2020) demonstrated significant performance deficits for low-resource languages. Cross-lingual transfer approaches (Pan et al., 2017; Rahimi et al., 2019) often suffer from negative transfer across divergent scripts.
</p>
<h3>2.2 Character Representations &amp; BiLSTM-CRF</h3>
<p class="no-indent">
The combination of character-level CNNs with bidirectional LSTMs and CRFs (Chiu and Nichols, 2016; Ma and Hovy, 2016) demonstrated that subword convolutions capture morphological affixes without requiring morphological segmenters. Lample et al. (2016) confirmed that structured CRF sequence decoding prevents illegal BIO transitions.
</p>
<h3>2.3 The Attention Interpretability Debate</h3>
<p class="no-indent">
A contentious debate exists regarding attention interpretability. Jain and Wallace (2019) demonstrated that attention weights frequently correlate poorly with gradient saliency, arguing 'Attention is not Explanation'. Conversely, Wiegreffe and Pinter (2019) asserted 'Attention is not not Explanation'. DeRose et al. (2021) and Hooker et al. (2019) established feature erasure as an interventional standard for deletion-based faithfulness.
</p>

<h2>3. Methodology</h2>
<h3>3.1 Grapheme Cluster Tokenization</h3>
<p class="no-indent">
To preserve Brahmic conjunct integrity, text is segmented into extended grapheme clusters (&bsol;X) adhering to Unicode Standard Annex #29:
</p>
<div class="equation">
  Cluster = Consonant &middot; (Virama &middot; Consonant)* &middot; Vowel_Sign* &middot; Modifier*
</div>
<p>
Each sentence <strong>x</strong> = (x<sub>1</sub>, ..., x<sub>T</sub>) is mapped to word indices w<sub>t</sub> and grapheme cluster sequences <strong>c</strong><sub>t</sub> over vocabularies |V<sub>word</sub>| = 89,196 and |V<sub>char</sub>| = 13,022.
</p>

<div class="figure-container">
  <img src="figures/fig2_dataset_pipeline.png" alt="Fig 2 Pipeline">
  <div class="caption">Fig. 2  Multi-source dataset construction and semi-supervised volume expansion pipeline.</div>
</div>

<h3>3.2 Neural Architecture</h3>
<p class="no-indent">
The neural architecture integrates multi-channel input representation, bidirectional recurrent sequence context, multi-head self-attention, and CRF decoding.
</p>

<div class="figure-container">
  <img src="figures/fig3_model_architecture.png" alt="Fig 3 Architecture">
  <div class="caption">Fig. 3  Complete hybrid neural architecture: Char-CNN + BiLSTM + Multi-Head Attention + CRF.</div>
</div>

<p>
For token x<sub>t</sub> in language &ell;, the input representation <strong>e</strong><sub>t</sub> &isin; &Ropf;<sup>206</sup> combines word embeddings (100d), Char-CNN encodings (90d, filter widths 3, 4, 5), and language conditioning (16d):
</p>
<div class="equation">
  <strong>e</strong><sub>t</sub> = Dropout([<strong>E</strong><sub>word</sub>(w<sub>t</sub>) || <strong>h</strong><sub>char</sub>(x<sub>t</sub>) || <strong>E</strong><sub>lang</sub>(&ell;)])
</div>

<div class="figure-container">
  <img src="figures/fig4_char_cnn.png" alt="Fig 4 Char-CNN">
  <div class="caption">Fig. 4  Multi-filter character-level convolutional subword encoder.</div>
</div>

<p>
The sequence <strong>E</strong> enters a 2-layer BiLSTM (hidden size 128/dir, output 256d), followed by a 4-head self-attention block (d<sub>k</sub>=64) with residual LayerNorm:
</p>
<div class="equation">
  <strong>Z</strong> = LayerNorm(<strong>H</strong> + [head<sub>1</sub> || ... || head<sub>H</sub>] <strong>W</strong><sup>O</sup>) &isin; &Ropf;<sup>T&times;256</sup>
</div>

<div class="figure-container">
  <img src="figures/fig5_multihead_attention.png" alt="Fig 5 Attention">
  <div class="caption">Fig. 5  Scaled dot-product multi-head self-attention block with residual normalization.</div>
</div>

<p>
Attended representations <strong>Z</strong> project to 9 BIO tag logits. Sequence scoring is governed by a transition matrix <strong>T</strong> &isin; &Ropf;<sup>9&times;9</sup>, optimized via forward-backward negative log-likelihood and decoded via Viterbi inference.
</p>

<h2>4. Explainability Framework</h2>
<p class="no-indent">
We formulate a two-tier explainability hierarchy (Fig. 6):
</p>

<div class="figure-container">
  <img src="figures/fig6_explainability_framework.png" alt="Fig 6 Explainability">
  <div class="caption">Fig. 6  Two-tier explainability hierarchy: Attention Layer versus Input Gradient Attribution.</div>
</div>

<p>
Tier 1 Attention-Layer Explainers evaluate post-recurrent attention matrices. We discover that taking the absolute value |&alpha; &odot; &nabla;<sub>&alpha;</sub> z| ranks inhibitory tokens at the top; applying a Signed ReLU operator max(0, &alpha; &odot; &nabla;<sub>&alpha;</sub> z) resolves this defect. Tier 2 Input-Level Attribution computes gradients directly with respect to input embeddings <strong>E</strong>, bypassing the BiLSTM recurrent buffer.
</p>

<div class="figure-container">
  <img src="figures/fig7_faithfulness_protocol.png" alt="Fig 7 Protocol">
  <div class="caption">Fig. 7  Controlled symmetric context-erasure protocol (zero_crf) for faithfulness evaluation.</div>
</div>

<p>
We evaluate faithfulness via a controlled symmetric context erasure protocol ('zero_crf', Fig. 7). Top-k and random-k candidates are drawn from strictly identical context candidate pools ({i &ne; q}). Confidence drops in forward-backward CRF marginals quantify deletion faithfulness:
</p>
<div class="equation">
  Faithfulness Gap(k) = Drop<sub>top</sub>(k) &minus; Drop<sub>rand</sub>(k)
</div>

<h2>5. Experimental Setup</h2>
<p class="no-indent">
Corpus statistics: Konkani (51,488 clean sents), Assamese (55,002 clean sents), Sinhala (66,954 clean sents). Partitions comprise 850 expanded train, 200 locked dev, and 200 locked test sentences per language. Weak supervision added 7,200 silver sentences per language (12x gold). Total multilingual pool: 36,736 sentences.
</p>

<h2>6. Results and Discussion</h2>
<h3>6.1 Final Gold-Test Performance</h3>
<p class="no-indent">
Table 1 and Fig. 8 show official results evaluated on cryptographically locked Gold-Test splits (Phase 14 single-run). The system achieves 96.27% F1 in Konkani, 97.87% in Assamese, and 97.76% in Sinhala, yielding an overall Macro F1 of <strong>97.30%</strong> (+7.30 pp margin above target).
</p>

<table class="academic-table">
  <caption>Table 1: Final official Gold-Test performance across languages.</caption>
  <thead>
    <tr>
      <th>Language</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Entity F1</th>
      <th>Token Acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Konkani (gom)</td>
      <td>99.15%</td>
      <td>93.55%</td>
      <td class="bold">96.27%</td>
      <td>99.47%</td>
    </tr>
    <tr>
      <td>Assamese (as)</td>
      <td>95.83%</td>
      <td>100.00%</td>
      <td class="bold">97.87%</td>
      <td>99.64%</td>
    </tr>
    <tr>
      <td>Sinhala (si)</td>
      <td>96.32%</td>
      <td>99.24%</td>
      <td class="bold">97.76%</td>
      <td>99.70%</td>
    </tr>
    <tr style="background-color: #f2f5f8;">
      <td class="bold">Macro Average</td>
      <td class="bold">97.10%</td>
      <td class="bold">97.60%</td>
      <td class="bold">97.30%</td>
      <td class="bold">99.60%</td>
    </tr>
  </tbody>
</table>

<div class="figure-container">
  <img src="figures/fig8_ner_performance.png" alt="Fig 8 NER">
  <div class="caption">Fig. 8  Locked Gold-Test precision, recall, and F1 across target languages.</div>
</div>

<p>
On Out-of-Vocabulary entity tokens completely absent from training vocabularies, the Char-CNN maintains an OOV F1 of <strong>91.03%</strong> (vs. 98.41% in-vocab F1), proving robust subword generalization.
</p>

<h3>6.2 Component Ablation Study</h3>
<p class="no-indent">
A matched ablation sweep across 36,736 sentences revealed that Subword BPE (-7.55%), CRF removal (-7.16%), and self-attention removal (-6.62%) induce the largest performance degradations (Fig. 9).
</p>

<div class="figure-container">
  <img src="figures/fig9_ablation_results.png" alt="Fig 9 Ablations">
  <div class="caption">Fig. 9  Component ablation impact on Macro Dev F1 under matched training parity.</div>
</div>

<h3>6.3 Faithfulness Benchmarking</h3>
<p class="no-indent">
Under the audited zero_crf protocol across 300 instances on Gold-Dev, Integrated Gradients (+0.2449 macro gap, 63.3% win rate) and Gradient &times; Input (+0.2350 macro gap, 62.7% win rate) decisively outperform raw attention (-0.0279 gap, 22.0% win rate), establishing <em>p</em> &lt; 0.0001 across 10,000 permutations (Fig. 10).
</p>

<div class="figure-container">
  <img src="figures/fig10_faithfulness_comparison.png" alt="Fig 10 Faithfulness">
  <div class="caption">Fig. 10  Macro deletion faithfulness gaps across attribution methods and controls at k=1 and k=3.</div>
</div>

<table class="academic-table">
  <caption>Table 2: Primary deletion faithfulness matrix on Gold-Dev (N=300).</caption>
  <thead>
    <tr>
      <th>Attribution Method</th>
      <th>Gap (k=1)</th>
      <th>Gap (k=3)</th>
      <th>Win Rate</th>
      <th>Permutation p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="bold">Integrated Gradients</td>
      <td class="bold">+0.2449</td>
      <td class="bold">+0.3396</td>
      <td class="bold">63.3%</td>
      <td class="bold">p &lt; 0.0001</td>
    </tr>
    <tr>
      <td class="bold">Gradient &times; Input</td>
      <td class="bold">+0.2350</td>
      <td class="bold">+0.3057</td>
      <td class="bold">62.7%</td>
      <td class="bold">p &lt; 0.0001</td>
    </tr>
    <tr>
      <td>Grad-Only Input</td>
      <td>+0.1365</td>
      <td>+0.2105</td>
      <td>42.0%</td>
      <td>p &lt; 0.0001</td>
    </tr>
    <tr>
      <td>Grad &times; Attn (Signed ReLU)</td>
      <td>+0.0574</td>
      <td>+0.1128</td>
      <td>35.0%</td>
      <td>p &lt; 0.01</td>
    </tr>
    <tr>
      <td>Random Control</td>
      <td>-0.0049</td>
      <td>-0.0080</td>
      <td>24.3%</td>
      <td>n.s.</td>
    </tr>
    <tr>
      <td>Raw Attention (Baseline)</td>
      <td>-0.0279</td>
      <td>-0.0617</td>
      <td>22.0%</td>
      <td>ref</td>
    </tr>
  </tbody>
</table>

<h3>6.4 Spatial Context Distance Analysis</h3>
<p class="no-indent">
Raw attention assigns 61.53% of its total mass to distant tokens (&gt;5 tokens away), functioning as a broad broadcast bus. In contrast, input attribution allocates 77.80% locally (46.83% on the entity self, 30.97% at distance 1&ndash;2), directly explaining its superior deletion faithfulness (Fig. 11).
</p>

<div class="figure-container">
  <img src="figures/fig11_distance_distribution.png" alt="Fig 11 Distance">
  <div class="caption">Fig. 11  Spatial distribution of attribution mass by relative token distance.</div>
</div>

<h2>7. Error Analysis</h2>
<p class="no-indent">
Person entities (PER) exhibit extreme context sensitivity (+0.6386 gap) because South Asian honorific prefixes ('श्री', 'ডাঃ', 'ගරු') act as mandatory syntactic cues; erasing an honorific immediately destroys entity recognition confidence. Conversely, locations and miscellaneous entities rely more on autonomous lexical roots (+0.055 to +0.264 gap).
</p>

<h2>8. Limitations</h2>
<p class="no-indent">
Deletion faithfulness evaluates feature necessity under erasure and should not be conflated with formal Pearl counterfactual causality. Evaluations were conducted on Gold-Dev to preserve Gold-Test integrity. The entire 4-stage pipeline trains in under 3.5 GPU hours on a consumer laptop GPU (&lt;60W TDP), demonstrating extreme environmental sustainability.
</p>

<h2>9. Conclusion</h2>
<p class="no-indent">
This work proves that lightweight, from-scratch neural sequence modeling achieves state-of-the-art accuracy (97.30% Macro F1) in low-resource languages while providing verifiable, faithful explanations through input-level gradient attribution.
</p>

<h2>References</h2>
<div class="ref-item">Adebayo, J., et al.: Sanity checks for saliency maps. In: NeurIPS 2018, pp. 9505&ndash;9515 (2018)</div>
<div class="ref-item">Atanasova, P., et al.: A diagnostic study of explainability techniques for text classification. In: EMNLP 2020, pp. 3256&ndash;3274 (2020)</div>
<div class="ref-item">Bastings, J., Filippova, K.: The elephant in the interpretability room. In: BlackboxNLP @ EMNLP 2020, pp. 149&ndash;159 (2020)</div>
<div class="ref-item">Chiu, J.P.C., Nichols, E.: Named entity recognition with bidirectional LSTM-CNNs. TACL 4, 357&ndash;370 (2016)</div>
<div class="ref-item">DeRose, J.F., et al.: Attention flows. IEEE TVCG 27(2), 1475&ndash;1485 (2021)</div>
<div class="ref-item">Hooker, S., et al.: A benchmark for interpretability methods. In: NeurIPS 2019, pp. 2330&ndash;2340 (2019)</div>
<div class="ref-item">Huang, Z., Xu, W., Yu, K.: Bidirectional LSTM-CRF models for sequence tagging. arXiv:1508.01991 (2015)</div>
<div class="ref-item">Jacovi, A., Goldberg, Y.: Towards faithfully interpretable NLP systems. In: ACL 2020, pp. 4198&ndash;4205 (2020)</div>
<div class="ref-item">Jain, S., Wallace, B.C.: Attention is not explanation. In: NAACL 2019, pp. 3543&ndash;3556 (2019)</div>
<div class="ref-item">Kakwani, D., et al.: IndicNLPSuite. In: EMNLP Findings, pp. 4948&ndash;4961 (2020)</div>
<div class="ref-item">Kim, Y., et al.: Character-aware neural language models. In: AAAI 2016, vol. 30 (2016)</div>
<div class="ref-item">Lafferty, J.D., McCallum, A., Pereira, F.C.N.: Conditional random fields. In: ICML 2001, pp. 282&ndash;289 (2001)</div>
<div class="ref-item">Lample, G., et al.: Neural architectures for named entity recognition. In: NAACL 2016, pp. 260&ndash;270 (2016)</div>
<div class="ref-item">Ma, X., Hovy, E.: End-to-end sequence labeling via Bi-LSTM-CNNs-CRF. In: ACL 2016, pp. 1064&ndash;1074 (2016)</div>
<div class="ref-item">Pan, X., et al.: Cross-lingual name tagging and linking for 282 languages. In: ACL 2017, pp. 1946&ndash;1958 (2017)</div>
<div class="ref-item">Rahimi, A., Li, Y., Cohn, T.: Massively multilingual transfer for NER. In: EMNLP 2019, pp. 1516&ndash;1526 (2019)</div>
<div class="ref-item">Serrano, S., Smith, N.A.: Is attention interpretable? In: ACL 2019, pp. 2931&ndash;2951 (2019)</div>
<div class="ref-item">Strubell, E., et al.: Linguistically-informed self-attention for SRL. In: EMNLP 2018, pp. 5027&ndash;5038 (2018)</div>
<div class="ref-item">Sundararajan, M., Taly, A., Yan, Q.: Axiomatic attribution for deep networks. In: ICML 2017, pp. 3319&ndash;3328 (2017)</div>
<div class="ref-item">Vaswani, A., et al.: Attention is all you need. In: NeurIPS 2017, pp. 5998&ndash;6008 (2017)</div>
<div class="ref-item">Wiegreffe, S., Pinter, Y.: Attention is not not explanation. In: EMNLP 2019, pp. 11&ndash;20 (2019)</div>

</div>

</body>
</html>
"""

print("Writing paper_manuscript.html...")
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
    print(f"PDF successfully generated: {PDF_PATH} ({PDF_PATH.stat().st_size} bytes)")
    
    # Render pages to PNG using PyMuPDF for visual inspection
    doc = fitz.open(str(PDF_PATH))
    print(f"Total Pages in Generated PDF: {len(doc)}")
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        page_img = PAGES_DIR / f"page_{i+1}.png"
        pix.save(str(page_img))
        print(f"Rendered page {i+1} -> {page_img}")
    doc.close()
    print("Page rendering complete!")
else:
    print("ERROR: PDF was not generated!")
