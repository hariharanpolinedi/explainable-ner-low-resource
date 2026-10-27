"""
Build Editable Microsoft Word Manuscript (.docx)
for 'Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution'
following Springer Nature Journal publication style.
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

doc = docx.Document()

# Page Setup: Standard Letter / A4, 1 inch margins
sections = doc.sections
for s in sections:
    s.top_margin = Inches(1.0)
    s.bottom_margin = Inches(1.0)
    s.left_margin = Inches(1.0)
    s.right_margin = Inches(1.0)

# Set base style font to Times New Roman
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(11)
font.color.rgb = RGBColor(0x22, 0x22, 0x22)

def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(18)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
    p.paragraph_format.space_after = Pt(12)

def add_authors(authors, affil):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p.add_run(authors + "\n")
    r1.font.size = Pt(12)
    r1.font.bold = True
    r2 = p.add_run(affil)
    r2.font.size = Pt(10)
    r2.font.italic = True
    p.paragraph_format.space_after = Pt(16)

def add_heading_1(text):
    h = doc.add_heading(level=1)
    run = h.add_run(text)
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
    h.paragraph_format.space_before = Pt(16)
    h.paragraph_format.space_after = Pt(6)

def add_heading_2(text):
    h = doc.add_heading(level=2)
    run = h.add_run(text)
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)

def add_p(text, italic=False, bold=False):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    run.font.italic = italic
    run.font.bold = bold
    return p

def add_callout(title, text):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    shading = parse_xml(r'<w:shd {} w:fill="F4F6F8"/>'.format(nsdecls('w')))
    cell._tc.get_or_add_tcPr().append(shading)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    r_t = p.add_run(title + "\n")
    r_t.font.bold = True
    r_t.font.size = Pt(10)
    r_d = p.add_run(text)
    r_d.font.size = Pt(9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def set_table_borders(table):
    tblPr = table._tbl.tblPr
    tblBorders = parse_xml(
        r'<w:tblBorders %s>'
        r'  <w:top w:val="single" w:sz="8" w:space="0" w:color="333333"/>'
        r'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="333333"/>'
        r'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>'
        r'  <w:left w:val="none"/>'
        r'  <w:right w:val="none"/>'
        r'  <w:insideV w:val="none"/>'
        r'</w:tblBorders>' % nsdecls('w')
    )
    tblPr.append(tblBorders)

def add_image_fig(path, caption, width=6.2):
    if Path(path).exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(10)
        p_img.paragraph_format.space_after = Pt(4)
        run = p_img.add_run()
        run.add_picture(str(path), width=Inches(width))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(12)
        r_cap = p_cap.add_run(caption)
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True
        r_cap.font.bold = True

print("Building Manuscript Word Document...")

# Title & Authors
add_title("Explainable Named Entity Recognition for Low-Resource Languages Using Attention Attribution")
add_authors("Author One and Author Two", "Department of Computer Science and Engineering, University Research Laboratory, India\nCorrespondence: author@university.edu")

# Abstract Box
abstract_text = (
    "Named Entity Recognition (NER) in morphologically rich, low-resource languages presents severe dual challenges: "
    "annotated supervision is scarce, and complex agglutinative orthographies break standard subword tokenization. "
    "Furthermore, dominant multilingual pretrained language models introduce prohibitive carbon footprints and unfaithful decision boundaries. "
    "In this work, we present a high-performance, strictly from-scratch neural NER framework for three low-resource South Asian languages: "
    "Konkani (gom, Devanagari script), Assamese (as, Eastern Nagari script), and Sinhala (si, Sinhala script). "
    "Constrained by a strict zero-pretrained discipline (no external weights, tokenizers, or embeddings), our architecture integrates character-level "
    "CNN grapheme cluster representations, bidirectional LSTM recurrence, multi-head self-attention, and a linear-chain Conditional Random Field (CRF). "
    "Trained on a phased multilingual pool combining weak supervision, confidence-filtered self-training, and targeted entity augmentation (36,736 sentences), "
    "the system achieves strict span-level F1 scores of 96.27% in Konkani, 97.87% in Assamese, and 97.76% in Sinhala on locked gold-test splits, "
    "yielding an overall macro F1 of 97.30% (exceeding the 90.0% research benchmark by +7.30 percentage points) with 91.03% out-of-vocabulary entity generalization. "
    "To explain model decisions, we investigate the faithfulness of internal attention versus input-level gradient attribution under a rigorously audited, "
    "target-isolated symmetric context erasure protocol. We show that raw self-attention directs 61.53% of its mass to distant sentence delimiters, "
    "acting as a broad syntactic broadcast bus rather than a faithful decision explainer (macro deletion gap: -0.0279). In contrast, bypassing the recurrent "
    "BiLSTM buffer via Gradient × Input and Integrated Gradients achieves decisive deletion faithfulness (+0.2350 and +0.2449 macro gaps, win rate 63.3%, "
    "p < 0.0001 across 10,000 permutations), concentrating over 78% of attribution on the entity and its immediate syntactic governors. Our findings establish "
    "that from-scratch neural sequence modeling can achieve state-of-the-art accuracy in low-resource settings while providing verifiable, faithful post-hoc explanations."
)
add_callout("Abstract", abstract_text)

p_kw = doc.add_paragraph()
r_kwt = p_kw.add_run("Keywords: ")
r_kwt.font.bold = True
p_kw.add_run("Named Entity Recognition, Low-Resource NLP, Explainable AI, Indic Languages, Multi-Head Self-Attention, Integrated Gradients, Brahmic Scripts, Faithfulness Evaluation.")
p_kw.paragraph_format.space_after = Pt(14)

# Section 1
add_heading_1("1. Introduction")
add_p(
    "Named Entity Recognition (NER) is a fundamental task in natural language processing (NLP), serving as an indispensable "
    "prerequisite for downstream information extraction, structured knowledge base construction, automated question answering, and "
    "neural machine translation. While recent paradigms relying on massive transformer-based multilingual pretrained language models "
    "(mPLMs) such as mBERT and XLM-RoBERTa have achieved impressive benchmarks in high-resource scenarios, extending these capabilities "
    "to low-resource languages remains an urgent scientific and societal challenge."
)
add_p(
    "South Asian and Indic languages present acute linguistic and computational hurdles. Languages such as Konkani (gom), Assamese (as), "
    "and Sinhala (si) are spoken by tens of millions of individuals across India and Sri Lanka, yet suffer from severe resource starvation "
    "in computational linguistics. Gold-standard expert-annotated corpora are exceptionally scarce, typically limited to fewer than 1,000 sentences. "
    "Crucially, these languages employ Brahmic script typologies (Devanagari, Eastern Nagari, and Sinhala script) characterized by complex "
    "consonant-vowel ligatures (aksharas), non-concatenative morphology, widespread homonymy, and extensive inflectional case suffixes "
    "(e.g., locative markers '-ंत', '-ত', and '-ට'). Standard subword tokenization schemes (e.g., Byte-Pair Encoding or WordPiece) "
    "routinely fracture these syllables across arbitrary byte boundaries, severing dependent vowel signs (matras) and destroying morphological integrity."
)
add_p(
    "To overcome these limitations, this research enforces a strict zero-pretrained discipline: zero external pretrained checkpoints, "
    "zero pretrained tokenizers, and zero external embeddings are utilized. Every component of our pipeline is engineered strictly from scratch. "
    "Furthermore, deploying neural taggers in real-world scenarios demands interpretability. We systematically investigate the faithfulness "
    "of internal self-attention versus input-level gradient attribution mechanisms, resolving long-standing controversies regarding attention "
    "interpretability in hybrid recurrent-attention sequence architectures."
)

add_image_fig("figures/fig1_overall_workflow.png", "Fig. 1  Comprehensive end-to-end research workflow from multi-source corpus harvesting to locked evaluation and explainability auditing.")

# Section 2
add_heading_1("2. Related Work")
add_heading_2("2.1 Named Entity Recognition in Low-Resource Indic Settings")
add_p(
    "NER in Indian and South Asian languages has historically suffered from sparse annotation benchmarks. Initiatives such as IndicGLUE "
    "(Kakwani et al., 2020) highlighted the performance gap between major scheduled languages (e.g., Hindi, Tamil) and peripheral tongues. "
    "Cross-lingual transfer methods (Pan et al., 2017; Rahimi et al., 2019) frequently encounter severe negative transfer when source and target "
    "scripts diverge typologically."
)
add_heading_2("2.2 Character-Level Context and BiLSTM-CRF")
add_p(
    "The integration of character-level convolutional neural networks with bidirectional LSTMs and Conditional Random Fields (Chiu and Nichols, 2016; "
    "Ma and Hovy, 2016) demonstrated that subword convolutions capture morphological affixes without requiring manual morphological analyzers. "
    "Lample et al. (2016) established that structured sequence decoding via linear-chain CRFs prevents illegal BIO transitions."
)
add_heading_2("2.3 The Attention Interpretability Debate and Faithfulness")
add_p(
    "Whether attention weights constitute faithful explanations has sparked intense debate. Jain and Wallace (2019) demonstrated that attention "
    "weights frequently correlate poorly with gradient saliency, concluding that 'Attention is not Explanation'. Conversely, Wiegreffe and Pinter (2019) "
    "argued that alternative attention distributions may fail under trained network constraints. DeRose et al. (2021) and Hooker et al. (2019) "
    "formalized feature erasure to evaluate deletion-based faithfulness."
)

# Section 3
add_heading_1("3. Methodology")
add_heading_2("3.1 Grapheme Cluster Tokenization")
add_p(
    "To preserve Brahmic syllable conjuncts, text is segmented into extended grapheme clusters conforming to Unicode Standard Annex #29: "
    "Cluster = Consonant · (Virama · Consonant)* · Vowel_Sign* · Modifier*. This preserves dependent matras and ligature integrity."
)

add_image_fig("figures/fig2_dataset_pipeline.png", "Fig. 2  Multi-source dataset construction and semi-supervised volume expansion pipeline.")

add_heading_2("3.2 Neural Architecture")
add_p(
    "The neural model combines Word Embeddings (100d), a multi-filter Character-Level CNN (90d, filter widths 3, 4, 5), and Language Conditioning (16d) "
    "into a 206-dimensional composite representation. This representation enters a 2-layer BiLSTM (hidden size 128/dir, output 256d), followed by a "
    "4-head Self-Attention layer (d_k=64) with residual LayerNorm, a linear emission layer, and a Linear-Chain CRF decoder with Viterbi decoding."
)

add_image_fig("figures/fig3_model_architecture.png", "Fig. 3  Complete hybrid neural architecture diagram showing layer dimensions and tensor shapes.")

add_image_fig("figures/fig4_char_cnn.png", "Fig. 4  Multi-filter character-level convolutional subword encoder architecture.")
add_image_fig("figures/fig5_multihead_attention.png", "Fig. 5  Scaled dot-product multi-head self-attention mechanism with residual normalization.")

# Section 4
add_heading_1("4. Explainability and Faithfulness Evaluation Framework")
add_p(
    "We formulate a two-tier explainability hierarchy: Tier 1 Attention-Layer Explainers (Raw Attention, Gradient × Attention, Signed ReLU) "
    "and Tier 2 Input-Level Attribution (Gradient × Input, Integrated Gradients). We discover that taking the absolute value in Gradient × Attention "
    "ranks inhibitory tokens at the top, creating an artificial negative gap artifact (-0.0277); applying a Signed ReLU operator resolves this defect."
)

add_image_fig("figures/fig6_explainability_framework.png", "Fig. 6  Two-tier explainability hierarchy: Attention Layer versus Input Gradient Attribution.")
add_image_fig("figures/fig7_faithfulness_protocol.png", "Fig. 7  Controlled symmetric context-erasure protocol (zero_crf) for faithfulness benchmarking.")

# Section 5 & 6
add_heading_1("5. Experimental Setup and Benchmark Partitions")
add_p(
    "Corpus statistics and partitions are verified as follows: Konkani (51,488 clean sents), Assamese (55,002 clean sents), Sinhala (66,954 clean sents). "
    "Gold splits comprise 850 expanded train, 200 locked dev, and 200 locked test sentences per language. Weak supervision added 7,200 silver sentences per language (12x gold)."
)

add_heading_1("6. Results and Discussion")
add_heading_2("6.1 Final Official Gold-Test Performance")
add_p(
    "Table 1 reports official single-run evaluations on locked Gold-Test splits. The system achieves 96.27% F1 in Konkani, 97.87% in Assamese, "
    "and 97.76% in Sinhala, yielding a Macro Entity F1 of 97.30% (+7.30 pp above the 90% benchmark) and token accuracy of 99.60%."
)

# Table 1 in Word
tbl_t1 = doc.add_table(rows=5, cols=6)
set_table_borders(tbl_t1)
tbl_t1.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_t1 = ["Language", "Precision", "Recall", "Entity F1", "Token Acc", "Margin vs >90%"]
for i, h in enumerate(headers_t1):
    cell = tbl_t1.cell(0, i)
    cell.paragraphs[0].add_run(h).font.bold = True
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

t1_rows = [
    ["Konkani (gom)", "99.15%", "93.55%", "96.27%", "99.47%", "+6.27 pp"],
    ["Assamese (as)", "95.83%", "100.00%", "97.87%", "99.64%", "+7.87 pp"],
    ["Sinhala (si)", "96.32%", "99.24%", "97.76%", "99.70%", "+7.76 pp"],
    ["Macro Average", "97.10%", "97.60%", "97.30%", "99.60%", "+7.30 pp"],
]
for r_idx, row in enumerate(t1_rows):
    for c_idx, val in enumerate(row):
        cell = tbl_t1.cell(r_idx + 1, c_idx)
        p = cell.paragraphs[0]
        run = p.add_run(val)
        if r_idx == 3 or c_idx == 3:
            run.font.bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph().paragraph_format.space_after = Pt(4)

add_image_fig("figures/fig8_ner_performance.png", "Fig. 8  Locked Gold-Test precision, recall, and F1 across target languages.")

add_heading_2("6.2 Component Ablation Study")
add_p(
    "A matched ablation sweep across 36,736 sentences revealed that Subword BPE (-7.55%), CRF removal (-7.16%), and self-attention removal (-6.62%) "
    "induce the largest macro performance drops."
)

add_image_fig("figures/fig9_ablation_results.png", "Fig. 9  Component ablation impact on Macro Dev F1 under matched training parity.")

add_heading_2("6.3 Attribution Faithfulness Benchmarking")
add_p(
    "Under the audited zero_crf protocol across 300 dev instances, Integrated Gradients (+0.2449 gap, 63.3% win rate) and Gradient × Input "
    "(+0.2350 gap, 62.7% win rate) decisively outperform raw attention (-0.0279 gap, 22.0% win rate), establishing p < 0.0001 across 10,000 permutations."
)

add_image_fig("figures/fig10_faithfulness_comparison.png", "Fig. 10  Macro deletion faithfulness gaps across attribution methods and controls at k=1 and k=3.")

add_heading_2("6.4 Why Attention Fails: Context Distance Allocation")
add_p(
    "Raw attention assigns 61.53% of its mass to distant tokens (>5 tokens away), acting as a broad syntactic broadcast bus. "
    "In contrast, input attribution allocates 77.80% locally (46.83% on the entity self, 30.97% on adjacent tokens at distance 1-2), explaining its superior deletion faithfulness."
)

add_image_fig("figures/fig11_distance_distribution.png", "Fig. 11  Spatial distribution of attribution mass by relative token distance.")

# Section 7, 8, 9
add_heading_1("7. Error Analysis and Qualitative Case Studies")
add_p(
    "Person names (PER) exhibit massive context sensitivity (+0.6386 gap) because South Asian honorific prefixes ('श्री', 'ডাঃ', 'ගරු') "
    "act as decisive syntactic anchors; erasing an honorific immediately destroys entity recognition confidence."
)

add_heading_1("8. Limitations and Epistemological Scope")
add_p(
    "We clarify that deletion-based faithfulness evaluates feature necessity under erasure, which involves out-of-distribution representation shifts, "
    "and should not be conflated with formal Pearl counterfactual causality. Evaluations were conducted on Gold-Dev to preserve Gold-Test integrity."
)

add_heading_1("9. Conclusion")
add_p(
    "This work proves that lightweight, from-scratch neural sequence modeling attains state-of-the-art accuracy (97.30% Macro F1) in low-resource "
    "languages while providing verifiable, faithful explanations through input-level gradient attribution."
)

# References
add_heading_1("References")
refs = [
    "Adebayo, J., Gilmer, J.D., Muelly, M., Goodfellow, I., Hardt, M., Kim, B.: Sanity checks for saliency maps. In: NeurIPS 2018, pp. 9505–9515 (2018)",
    "Atanasova, P., Simonsen, J.G., Lioma, C., Augenstein, I.: A diagnostic study of explainability techniques for text classification. In: EMNLP 2020, pp. 3256–3274 (2020)",
    "Bastings, J., Filippova, K.: The elephant in the interpretability room: Why why attention is not explanation. In: BlackboxNLP @ EMNLP 2020, pp. 149–159 (2020)",
    "Chiu, J.P.C., Nichols, E.: Named entity recognition with bidirectional LSTM-CNNs. Trans. Assoc. Comput. Linguist. 4, 357–370 (2016)",
    "DeRose, J.F., et al.: Attention flows: Analyzing and comparing attention mechanisms in language models. IEEE TVCG 27(2), 1475–1485 (2021)",
    "Hooker, S., Erhan, D., Kindermans, P.J., Kim, B.: A benchmark for interpretability methods. In: NeurIPS 2019, pp. 2330–2340 (2019)",
    "Huang, Z., Xu, W., Yu, K.: Bidirectional LSTM-CRF models for sequence tagging. arXiv preprint arXiv:1508.01991 (2015)",
    "Jacovi, A., Goldberg, Y.: Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness? In: ACL 2020, pp. 4198–4205 (2020)",
    "Jain, S., Wallace, B.C.: Attention is not explanation. In: NAACL 2019, pp. 3543–3556 (2019)",
    "Kakwani, D., et al.: IndicNLPSuite: Monolingual corpora, evaluation benchmarks and pre-trained multilingual language models for Indian languages. In: EMNLP Findings, pp. 4948–4961 (2020)",
    "Kim, Y., Jernite, Y., Sontag, D., Rush, A.M.: Character-aware neural language models. In: AAAI 2016, vol. 30 (2016)",
    "Lafferty, J.D., McCallum, A., Pereira, F.C.N.: Conditional random fields: Probabilistic models for segmenting and labeling sequence data. In: ICML 2001, pp. 282–289 (2001)",
    "Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., Dyer, C.: Neural architectures for named entity recognition. In: NAACL 2016, pp. 260–270 (2016)",
    "Ma, X., Hovy, E.: End-to-end sequence labeling via bi-directional LSTM-CNNs-CRF. In: ACL 2016, pp. 1064–1074 (2016)",
    "Pan, X., Zhang, B., May, J., Nothman, J., Knight, K., Ji, H.: Cross-lingual name tagging and linking for 282 languages. In: ACL 2017, pp. 1946–1958 (2017)",
    "Rahimi, A., Li, Y., Cohn, T.: Massively multilingual transfer for NER using global constraints. In: EMNLP 2019, pp. 1516–1526 (2019)",
    "Serrano, S., Smith, N.A.: Is attention interpretable? In: ACL 2019, pp. 2931–2951 (2019)",
    "Strubell, E., Verga, P., Andor, D., Weiss, D., McCallum, A.: Linguistically-informed self-attention for semantic role labeling. In: EMNLP 2018, pp. 5027–5038 (2018)",
    "Sundararajan, M., Taly, A., Yan, Q.: Axiomatic attribution for deep networks. In: ICML 2017, pp. 3319–3328 (2017)",
    "Vaswani, A., et al.: Attention is all you need. In: NeurIPS 2017, pp. 5998–6008 (2017)",
    "Wiegreffe, S., Pinter, Y.: Attention is not not explanation. In: EMNLP 2019, pp. 11–20 (2019)"
]

for r in refs:
    p_r = doc.add_paragraph()
    p_r.paragraph_format.left_indent = Inches(0.25)
    p_r.paragraph_format.first_line_indent = Inches(-0.25)
    p_r.paragraph_format.space_after = Pt(4)
    run = p_r.add_run(r)
    run.font.size = Pt(9)

out_docx = Path("Explainable_NER_Low_Resource_Languages_Attention_Attribution.docx")
doc.save(str(out_docx))
print(f"Successfully generated {out_docx} ({out_docx.stat().st_size} bytes)")
