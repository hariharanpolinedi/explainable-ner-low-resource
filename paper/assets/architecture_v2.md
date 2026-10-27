# Final Model Architecture Specification (v2 System)

## 1. Architectural Philosophy & Design Principles

The v2 Explainable Named Entity Recognition (NER) architecture is designed for **high-reliability low-resource sequence tagging** across typologically diverse languages. It reconciles two conflicting objectives:
1. **Multilingual Shared Representation Learning**: Maximizing positive inductive transfer from high-resource and moderately-resourced languages to ultra-low-resource languages.
2. **Strict Interference Isolation**: Guaranteeing that the integration of new languages (such as Kokborok) or targeted retraining (such as Reliability Cycle 2 for Konkani and Assamese) causes **zero degradation** to established language performance.

To achieve this, the v2 system adopts a **Staged Modular Multi-Head Attention BiLSTM-CRF Architecture** with language-conditioned sequence tagging heads and frozen backbone adaptation.

---

## 2. High-Level Architecture Diagram

```
Input Sequence: [ w_1, w_2, ..., w_T ]  (Language ID: l in {kok, gom, asm, sin})
      │
      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   SHARED MULTILINGUAL ENCODER TRUNK                   │
│                                                                        │
│  ┌───────────────────────┐          ┌───────────────────────────────┐  │
│  │   Word Embedding      │          │     Character-CNN Module      │  │
│  │   (d_word = 100)      │          │  (Filters: 25, 50; d_char=30) │  │
│  └───────────┬───────────┘          └───────────────┬───────────────┘  │
│              └───────────────┬──────────────────────┘                  │
│                              ▼                                         │
│                 Fused Token Representation e_t                         │
│                              │                                         │
│                              ▼                                         │
│             Bidirectional LSTM Layer (BiLSTM)                          │
│             (Hidden Dim: 2 x 128 = 256, 1 Layer)                       │
│                              │                                         │
│                              ▼                                         │
│           Multi-Head Self-Attention Layer (MHSA)                       │
│           (Heads: 4, Key/Query/Value Dim: 64, Out: 256)                │
│                              │                                         │
│                              ▼                                         │
│             Contextual Hidden Representation h_t                       │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
┌────────────────────────────────┐   ┌──────────────────────────────────┐
│ MODULAR CRF TAGGING HEADS      │   │ EXPLAINABILITY & ATTRIBUTION     │
│                                │   │                                  │
│ • Linear Emission Projection   │   │ • Attention Attribution Matrix   │
│   W_e^(l) * h_t + b_e^(l)      │   │   A = softmax(Q K^T / sqrt(d_k)) │
│ • Language Transition Matrix   │   │ • Gradient-Weighted Attribution  │
│   A_CRF^(l) in R^(K x K)       │   │   Grad x Attn = A * |grad_A|     │
│ • Viterbi Decoding Inference   │   │ • Symmetric Erasure Faithfulness │
│   y* = argmax Score(x, y; l)   │   │   Delta P(exclude target token)  │
└────────────────────────────────┘   └──────────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### 3.1 Input Embedding Layer
- **Word Embeddings**: Each token $w_t$ is mapped to a dense vector $\mathbf{v}_t \in \mathbb{R}^{d_w}$ with dimension $d_w = 100$. Vocabulary is constructed across all languages with shared special tokens (`<PAD>`, `<UNK>`).
- **Character-CNN Sub-word Module**: To handle morphologically complex and out-of-vocabulary (OOV) tokens, each character $c_{t,j}$ in token $w_t$ is embedded in $\mathbb{R}^{d_c}$ ($d_c = 30$). A 1D convolutional filter bank with kernel widths $k \in \{3, 5\}$ and filter counts $f_k \in \{25, 50\}$ extracts n-gram character features, followed by Max-Over-Time Pooling:
  $$\mathbf{u}_t = \text{MaxPool}\left(\text{ReLU}\left(\text{Conv1D}(\mathbf{C}_t)\right)\right) \in \mathbb{R}^{75}$$
- **Fused Token Representation**:
  $$\mathbf{e}_t = \left[ \mathbf{v}_t \,\|\, \mathbf{u}_t \right] \in \mathbb{R}^{175}$$

### 3.2 Bidirectional Recurrent Layer (BiLSTM)
The sequence of fused embeddings $\mathbf{e}_{1:T}$ is processed by a bidirectional LSTM:
$$\overrightarrow{\mathbf{h}}_t = \text{LSTM}_{fwd}(\mathbf{e}_t, \overrightarrow{\mathbf{h}}_{t-1}), \quad \overleftarrow{\mathbf{h}}_t = \text{LSTM}_{bwd}(\mathbf{e}_t, \overleftarrow{\mathbf{h}}_{t+1})$$
$$\mathbf{h}_t^{LSTM} = \left[ \overrightarrow{\mathbf{h}}_t \,\|\, \overleftarrow{\mathbf{h}}_t \right] \in \mathbb{R}^{256} \quad (\text{forward: } 128, \text{ backward: } 128)$$

### 3.3 Multi-Head Self-Attention Layer (MHSA)
To capture non-local syntactic dependencies and provide direct, interpretable token-to-token importance weights, the BiLSTM hidden states are passed through a Multi-Head Self-Attention mechanism:
$$\mathbf{Q} = \mathbf{H} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H} \mathbf{W}_V$$
$$\mathbf{A}_m = \text{softmax}\left( \frac{\mathbf{Q}_m \mathbf{K}_m^T}{\sqrt{d_k}} \right) \in \mathbb{R}^{T \times T}$$
$$\text{head}_m = \mathbf{A}_m \mathbf{V}_m$$
$$\mathbf{H}^{Attn} = \left[ \text{head}_1 \,\|\, \dots \,\|\, \text{head}_M \right] \mathbf{W}_O \in \mathbb{R}^{T \times 256}$$
- **Hyperparameters**: Number of heads $M = 4$, Key/Query dimension $d_k = 64$, Total projected dimension $d_model = 256$, Dropout $p = 0.33$.
- **Residual Connection & Layer Norm**:
  $$\mathbf{H}^{out} = \text{LayerNorm}(\mathbf{H}^{LSTM} + \mathbf{H}^{Attn})$$

### 3.4 Modular Language-Conditioned CRF Heads
Rather than forcing all languages through a single monolithic emission matrix and transition graph, the v2 system implements **language-conditioned sequence heads**:
- **Language Emission Projection**:
  $$\mathbf{E}_t^{(l)} = \mathbf{W}_e^{(l)} \mathbf{h}_t^{out} + \mathbf{b}_e^{(l)} \in \mathbb{R}^{K}$$
  where $K$ is the number of BIO tags (e.g., 9 for CoNLL-4: $O, B\text{-}PER, I\text{-}PER, \dots$), and $l \in \{\text{gom}, \text{asm}, \text{sin}, \text{kok}\}$.
- **Linear-Chain Conditional Random Field (CRF)**:
  For a candidate sequence $\mathbf{y} = (y_1, \dots, y_T)$, the path score under language $l$ is:
  $$\text{Score}(\mathbf{x}, \mathbf{y}; l) = \sum_{t=1}^T \mathbf{E}_{t, y_t}^{(l)} + \sum_{t=1}^{T-1} \mathbf{T}_{y_t, y_{t+1}}^{(l)}$$
  where $\mathbf{T}^{(l)} \in \mathbb{R}^{K \times K}$ is the language-specific transition matrix.
- **Decoding**: Exact global sequence prediction is performed using the Viterbi algorithm:
  $$\mathbf{y}^* = \arg\max_{\mathbf{y}} \text{Score}(\mathbf{x}, \mathbf{y}; l)$$

---

## 4. Architectural Modifications Across Development Tracks

### 4.1 Reliability Cycle 2 (Konkani & Assamese)
- **Architectural Invariance**: It is critical to emphasize that **Reliability Cycle 2 did NOT modify the neural architecture structure**. The BiLSTM, Char-CNN, and Attention dimensionalities remained exactly identical.
- **Modifications Applied**:
  1. *Training Data Reconciliation*: Cleaned 699 erroneous annotations in the Assamese corpus.
  2. *Loss Function Weighting*: Implemented an 8x entity penalty multiplier during negative log-likelihood calculation for Assamese non-$O$ tokens.
  3. *Transition Matrix Recalibration*: Recalibrated the transition prior $\mathbf{T}_{B \rightarrow O}^{(asm)}$ and $\mathbf{T}_{B \rightarrow I}^{(asm)}$, slashing background $O$-class emission dominance by $>91\%$.
  4. *Protected Checkpoints*: Produced `current_best_konkani.pt` and `current_best_assamese.pt`.

### 4.2 Phase 12 / 13 Integration Strategy (Kokborok)
- **Staged Frozen-Backbone Adaptation**:
  - The shared multilingual trunk ($\text{Embedding} + \text{Char-CNN} + \text{BiLSTM} + \text{Attention}$) trained on the original 3 languages was **completely frozen** ($\nabla_{\theta_{trunk}} = 0$).
  - A new modular emission projection $\mathbf{W}_e^{(kok)}$ and transition matrix $\mathbf{T}^{(kok)}$ were initialized and trained exclusively on Kokborok training data.
  - This prevented any gradient backpropagation from Kokborok into the shared trunk, guaranteeing **0.00 pp performance regression** on the original 3 languages while providing Kokborok with rich multilingual contextual features.
- **Checkpoint**: `checkpoints/final_kokborok.pt`.

---

## 5. Explainability Architecture & Faithfulness Protocol

### 5.1 Attribution Extraction
For an entity span spanning tokens $[t_{start}, t_{end}]$, explainability attribution is computed relative to all other context tokens $j \neq [t_{start}, t_{end}]$:
1. **Raw Attention Attribution**:
   $$\alpha_j = \frac{1}{M} \sum_{m=1}^M \frac{1}{|E|} \sum_{i \in E} \mathbf{A}_{m, i, j}$$
2. **Gradient $\times$ Attention Attribution**:
   $$\gamma_j = \alpha_j \cdot \left| \frac{\partial \mathcal{L}_{NER}}{\partial \alpha_j} \right|$$

### 5.2 Symmetric Erasure Faithfulness Protocol
To validate which attribution method faithfully reflects the model's decision logic, the **Symmetric Erasure Test** is integrated into the evaluation pipeline:
- **Rule**: For both Top-$k$ attributed tokens and Random-$k$ baseline tokens, the target entity tokens themselves are **strictly excluded** from the candidate erasure pool (`exclude_target_token=True`).
- **Metric**: Faithfulness is measured by the prediction probability drop upon erasing top-context tokens versus random tokens:
  $$\Delta P_{drop}(k) = \left( P(y^* \mid \mathbf{x}) - P(y^* \mid \mathbf{x}_{\setminus \text{top-}k}) \right) - \left( P(y^* \mid \mathbf{x}) - P(y^* \mid \mathbf{x}_{\setminus \text{rand-}k}) \right)$$
- A positive gap ($\Delta P_{drop} > 0$) confirms that the attribution method successfully identifies the tokens most crucial to the model's classification confidence.

---

## 6. Architecture Summary Table

| Parameter / Module | Value / Configuration | Scope |
|:---|:---|:---|
| Word Embedding Dimension | $100$ | Shared Multilingual |
| Char Embedding Dimension | $30$ | Shared Multilingual |
| Char-CNN Kernels | $[3, 5]$, Filters: $[25, 50]$ | Shared Multilingual |
| BiLSTM Hidden Dimension | $256$ ($128$ fwd + $128$ bwd) | Shared Multilingual |
| BiLSTM Layers | $1$ | Shared Multilingual |
| Self-Attention Heads ($M$) | $4$ | Shared Multilingual |
| Attention Head Dimension ($d_k$) | $64$ (Total: $256$) | Shared Multilingual |
| Dropout Rate | $0.33$ | Shared Multilingual |
| Language Tagging Heads | $4$ independent CRF heads | Modular per Language |
| Loss Function | CRF Negative Log-Likelihood | Language-Conditioned |
| Decoding Algorithm | Exact Viterbi Decoding | Per-Language Runtime |
