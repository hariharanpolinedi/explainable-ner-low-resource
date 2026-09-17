# Methodology: Explainable Named Entity Recognition for Low-Resource Languages

## 1. Introduction and Problem Formulation

Named Entity Recognition (NER) in morphologically rich, low-resource languages poses fundamental challenges for standard neural sequence modeling. In languages such as Konkani (Devanagari script), Assamese (Bengali-Assamese script), and Sinhala (Sinhala script), labeled training corpora are severely constrained ($\le 1,000$ gold sentences per language), while orthographic complexities—including complex consonant clusters (aksharas), non-concatenative inflectional suffixes, and widespread polysemy—preclude naive transfer from high-resource benchmarks. 

Furthermore, standard industrial paradigms relying on massive pretrained language models (e.g., mBERT, XLM-RoBERTa) introduce significant carbon footprints, vocabulary truncation on low-resource scripts, and uninterpretable decision boundaries. In this project, we establish an absolute constraint: **zero external pretrained models, tokenizers, or embeddings are utilized**. Every component of the feature pipeline, neural architecture, semi-supervised data expansion, and explainability mechanism is constructed and trained strictly from scratch.

Formally, given an input sentence $\mathbf{x} = (x_1, x_2, \dots, x_T)$ in language $\ell \in \{\text{Konkani}, \text{Assamese}, \text{Sinhala}\}$, the NER task requires predicting an optimal sequence of categorical labels $\mathbf{y} = (y_1, y_2, \dots, y_T)$ under a 9-class BIO schema:
$$\mathcal{Y} = \{\text{O}, \text{B-PER}, \text{I-PER}, \text{B-LOC}, \text{I-LOC}, \text{B-ORG}, \text{I-ORG}, \text{B-MISC}, \text{I-MISC}\}$$
Beyond sequence labeling accuracy, we require the model to emit faithful token-level attribution weights $\mathbf{a}_t \in \mathbb{R}^T$ explaining each entity prediction $y_t \in \mathcal{Y} \setminus \{\text{O}\}$ without external black-box surrogate explainer models.

---

## 2. Script-Aware Preprocessing and Orthographic Representation

### 2.1 Unicode Normalization and Conjunct Integrity
Low-resource South Asian languages frequently suffer from encoding variance, orphaned viramas (halants), and inconsistent zero-width joiner (ZWJ / U+200D) or non-joiner (ZWNJ / U+200C) sequences. The preprocessing pipeline applies strict Unicode Normalization Form C (NFC). In Sinhala, where ZWJ sequences dictate whether a consonant cluster forms a touching conjunct (bandi akshara) or a yansaya/rakaaraya ligature, joiners are strictly preserved within valid orthographic consonant-virama-ZWJ clusters and stripped only when orphaned or trailing whitespace.

### 2.2 Extended Grapheme Cluster Tokenization
Standard whitespace and byte-pair subword tokenizers frequently fracture Indic syllable blocks across arbitrary byte boundaries, severing dependent vowel signs (matras) from base consonants. To prevent morphological destruction, we segment words into extended grapheme clusters ($\backslash\text{X}$) conforming to Unicode Standard Annex #29:
$$\text{cluster} = \text{Consonant} \cdot (\text{Virama} \cdot \text{Consonant})^* \cdot \text{Vowel\_Sign}^* \cdot \text{Anusvara}^*$$
Each token $x_t$ is mapped to:
1. A categorical word index $w_t \in \{1, \dots, |\mathcal{V}_{\text{word}}|\}$, where $|\mathcal{V}_{\text{word}}| = 89,196$ pooled across all three languages.
2. A sequence of grapheme cluster indices $\mathbf{c}_t = (c_{t,1}, c_{t,2}, \dots, c_{t,L_t})$ over a character/grapheme vocabulary $|\mathcal{V}_{\text{char}}| = 13,022$.

---

## 3. Neural Model Architecture

The core predictor combines three complementary inductive biases: subword morphological extraction, bidirectional recurrent sequence context, and multi-head self-attention with linear-chain sequence decoding.

```
                  ┌────────────────────────────────────────┐
                  │    Linear-Chain CRF Sequence Decoder   │
                  └────────────────────────────────────────┘
                                      ▲
                                      │ (Attended Emissions)
                  ┌────────────────────────────────────────┐
                  │    Multi-Head Self-Attention Layer     │  ◄── Attn Instrumentation
                  │   (H=4 heads, d_k=64, Residual + LN)   │      (Weights Exposed)
                  └────────────────────────────────────────┘
                                      ▲
                                      │ (Contextual Hidden States)
                  ┌────────────────────────────────────────┐
                  │     Bidirectional LSTM Encoder         │
                  │   (Hidden Dim = 128, BiLSTM Out = 256) │
                  └────────────────────────────────────────┘
                                      ▲
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
   ┌─────────────────────┐   ┌──────────────────────┐  ┌──────────────────────┐
   │ 100d Word Embedding │   │ 90d Char-CNN Encoder │  │ 16d Lang Embedding   │
   │ (Learned from 0)    │   │ (Filters: 3, 4, 5)   │  │ (Learned from 0)     │
   └─────────────────────┘   └──────────────────────┘  └──────────────────────┘
```

### 3.1 Composite Input Embeddings
For token $x_t$ in language $\ell$, the input representation $\mathbf{e}_t \in \mathbb{R}^{206}$ is defined by:
$$\mathbf{e}_t = \text{Dropout}\Big( \big[ \mathbf{E}_{\text{word}}(w_t) \,\|\, \mathbf{h}_{\text{char}}(x_t) \,\|\, \mathbf{E}_{\text{lang}}(\ell) \big] \Big)$$
where:
- $\mathbf{E}_{\text{word}} \in \mathbb{R}^{|\mathcal{V}_{\text{word}}| \times 100}$ represents randomly initialized word lexical embeddings.
- $\mathbf{E}_{\text{lang}} \in \mathbb{R}^{3 \times 16}$ represents a dedicated language conditioning vector modulating lexical inputs by script typology.
- $\mathbf{h}_{\text{char}}(x_t) \in \mathbb{R}^{90}$ is the character-level CNN encoding.

The Char-CNN embeds individual grapheme clusters into 50-dimensional vectors, processes them through parallel 1D convolution banks with filter widths $k \in \{3, 4, 5\}$ (30 filters per width), applies ReLU activation, and pools across time:
$$h_j^{(k)} = \max_{1 \le i \le L_t - k + 1} \text{ReLU}\big( \mathbf{W}_k * \mathbf{c}_{t, i:i+k-1} + b_k \big)$$
$$\mathbf{h}_{\text{char}}(x_t) = \big[ \mathbf{h}^{(3)} \,\|\, \mathbf{h}^{(4)} \,\|\, \mathbf{h}^{(5)} \big] \in \mathbb{R}^{90}$$
This allows the network to natively capture inflectional prefixes (e.g., honorific markers `ড°`, `डॉ.`, `ගරු`), locative postpositional case endings (e.g., `-ंत`, `-ৰ`, `-ට`), and derivational affixes without subword fragmentation.

### 3.2 Bidirectional Contextual Encoder
The sequence of composite embeddings $\mathbf{E} = (\mathbf{e}_1, \dots, \mathbf{e}_T)$ is fed to a bidirectional LSTM:
$$\overrightarrow{\mathbf{h}}_t = \text{LSTM}_{\text{fwd}}(\mathbf{e}_t, \overrightarrow{\mathbf{h}}_{t-1}), \quad \overleftarrow{\mathbf{h}}_t = \text{LSTM}_{\text{bwd}}(\mathbf{e}_t, \overleftarrow{\mathbf{h}}_{t+1})$$
$$\mathbf{h}_t = \big[ \overrightarrow{\mathbf{h}}_t \,\|\, \overleftarrow{\mathbf{h}}_t \big] \in \mathbb{R}^{256}$$

### 3.3 Multi-Head Self-Attention Layer
To model long-range syntactic agreement and resolve multi-word entity dependencies across distant clauses, contextual states $\mathbf{H} \in \mathbb{R}^{T \times 256}$ are passed into a scaled multi-head self-attention layer with $H = 4$ heads ($d_k = 64$):
$$\mathbf{Q}_h = \mathbf{H} \mathbf{W}_h^Q, \quad \mathbf{K}_h = \mathbf{H} \mathbf{W}_h^K, \quad \mathbf{V}_h = \mathbf{H} \mathbf{W}_h^V$$
$$\mathbf{A}_h = \text{softmax}\left( \frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d_k}} + \mathbf{M} \right) \in \mathbb{R}^{T \times T}$$
$$\text{head}_h = \mathbf{A}_h \mathbf{V}_h \in \mathbb{R}^{T \times 64}$$
where $\mathbf{M}$ is an exact numerical padding mask ($M_{i,j} = -\infty$ if token $j$ is padding, enforcing $\mathbf{A}_{h, i, \text{pad}} \equiv 0$). The concatenated head representations are projected, combined via residual connection, and layer-normalized:
$$\mathbf{Z} = \text{LayerNorm}\Big( \mathbf{H} + \big[ \text{head}_1 \,\|\, \dots \,\|\, \text{head}_H \big] \mathbf{W}^O \Big) \in \mathbb{R}^{T \times 256}$$

### 3.4 Linear-Chain Conditional Random Field (CRF)
Raw token-level emission potentials $\mathbf{P} \in \mathbb{R}^{T \times 9}$ are projected from attended representations: $\mathbf{P} = \mathbf{Z} \mathbf{W}^E$. To penalize invalid tag transitions (e.g., `O` $\rightarrow$ `I-PER`), sequence scoring is governed by a transition matrix $\mathbf{T} \in \mathbb{R}^{9 \times 9}$:
$$s(\mathbf{x}, \mathbf{y}) = \sum_{t=1}^T P_{t, y_t} + \sum_{t=1}^{T-1} T_{y_t, y_{t+1}}$$
The conditional log-likelihood of the gold label sequence $\mathbf{y}^*$ is maximized during training via the forward-backward partition function $Z(\mathbf{x})$:
$$\mathcal{L}_{\text{CRF}}(\theta) = -\log p(\mathbf{y}^* \mid \mathbf{x}) = -\left( s(\mathbf{x}, \mathbf{y}^*) - \log \sum_{\mathbf{y}' \in \mathcal{Y}^T} \exp\big(s(\mathbf{x}, \mathbf{y}')\big) \right)$$
At inference time, exact global decoding is computed via the Viterbi dynamic program:
$$\hat{\mathbf{y}} = \arg\max_{\mathbf{y} \in \mathcal{Y}^T} s(\mathbf{x}, \mathbf{y})$$

---

## 4. Multi-Stage Training and Data Augmentation Pipeline

Training follows a phased progression designed to counteract resource starvation:

```
┌─────────────────────────┐
│ Phase 1: Raw Corpora    │  Wikipedia, Press, Public Domain Archives
│ (Konkani, Assamese,     │  (51.5k Konkani, 55.0k Assamese, 67.0k Sinhala)
│  Sinhala Text Pools)    │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Phase 4: Weak Superv.   │  Gazetteers + Toponymic Rules + Wiki Distant Supervision
│ (Silver Data Assembly)  │  7,200 silver sents/lang at >=96% Precision
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Phase 8: Joint Warmup   │  Multilingual Pooled Training with 16d Language Embedding
│ (Multilingual Pretrain) │  (Shared BiLSTM + Multi-Head Attention, AdamW lr=1e-3)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Phase 9: Specialization │  Per-Language Fine-Tuning with Layer-Unfreezing Schedule
│ (Target Fine-Tuning)    │  (Lower-layer parameter freezing, lr=2e-4)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Phase 10: Self-Training │  CRF Sequence Confidence Filtering (tau >= 0.90 -> 0.92)
│ (Pseudo-Label Loop)     │  Iterative re-training with rollback on regression
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Phase 11: Augmentation  │  Gazetteer Entity Substitution + Class-Weighted Loss
│ (Rare Entity Balancing) │  Targeted expansion of underrepresented ORG / LOC classes
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Phase 12: Final Lock    │  Systematic Hyperparameter Search & Master Ablation Suite
│ (Production Model)      │  Configs and Checkpoints Frozen for Phase 13–15
└─────────────────────────┘
```

1. **Multilingual Joint Warm-Start**: All 23,400 gold and silver sentences across Konkani, Assamese, and Sinhala are pooled into a single multi-task training corpus. A shared encoder is optimized using uniform batch sampling across languages. This forces the lower BiLSTM layers to learn cross-script phonological and syntactic invariants (e.g., SOV word order shared across Indic and Sinhala grammars).
2. **Layer-Unfreezing Specialization**: Per-language fine-tuning initializes from the joint warm-start checkpoint. In initial epochs, lower embedding and BiLSTM layers remain frozen while the upper attention projection and CRF transition matrices adapt to script-specific vocabulary distributions. Subsequent epochs unfreeze all parameters with a reduced learning rate ($2 \times 10^{-4}$).
3. **Iterative Self-Training with Rollback**: Unlabeled raw sentences are evaluated under the fine-tuned model. Sentences whose tokens satisfy strict sequence-level CRF confidence thresholds ($\tau \ge 0.90$, tightening to $0.92$ in round 2) are admitted to the training pool. Any iteration exhibiting dev-set regression is automatically rolled back to the previous checkpoint.
4. **Targeted Gazetteer Augmentation**: Synthetic entity substitution swaps known gazetteer entities into existing training frames, expanding lexical diversity while preserving syntactic context.

---

## 5. Attention Attribution and Faithfulness Validation

### 5.1 Gradient × Attention Computation
To explain the classification of entity token $x_t$ with predicted tag $\hat{y}_t$, we compute the sensitivity of the emission score $P_{t, \hat{y}_t}$ with respect to the attention weight tensor $\mathbf{A} \in \mathbb{R}^{H \times T \times T}$:
$$\mathbf{G}_{h, t, j} = \frac{\partial P_{t, \hat{y}_t}}{\partial A_{h, t, j}}$$
The token attribution score for context position $j$ is computed by the element-wise Hadamard product, averaged over all attention heads $h \in \{1, \dots, H\}$, and normalized over sequence length:
$$S(j \mid t) = \frac{1}{H} \sum_{h=1}^H \big| G_{h, t, j} \cdot A_{h, t, j} \big|$$
$$\tilde{S}(j \mid t) = \frac{S(j \mid t)}{\sum_{k=1}^T S(k \mid t)}$$

### 5.2 Erasure Faithfulness Criterion
Attribution explanations are validated using a rigorous token erasure protocol evaluated over $\ge 100$ gold-dev predictions per language. For each entity prediction, the top-$k$ attributed context tokens ($k \in \{1, 2, 3\}$) are masked to the out-of-vocabulary marker `"<UNK>"`, and the resulting decrease in predicted tag confidence is measured:
$$\Delta \text{Conf}_{\text{top-}k} = p(\hat{y}_t \mid \mathbf{x}) - p(\hat{y}_t \mid \mathbf{x}_{\setminus \text{top-}k})$$
As a strict control, $k$ random non-top tokens are erased:
$$\Delta \text{Conf}_{\text{rand-}k} = p(\hat{y}_t \mid \mathbf{x}) - p(\hat{y}_t \mid \mathbf{x}_{\setminus \text{rand-}k})$$
The **Faithfulness Gap** is defined as:
$$\text{Gap}(k) = \Delta \text{Conf}_{\text{top-}k} - \Delta \text{Conf}_{\text{rand-}k}$$
Comparing Gradient $\times$ Attention against Raw Attention empirically demonstrates whether scaling attention by emission gradients filters out uninformative syntactic delimiters and grounds explanations in actual decision causality.
