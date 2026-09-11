# ⚡ DeepTrace AI — Multi-Kernel 1D-CNN AI Text Detection Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![HuggingFace](https://img.shields.io/badge/BERT-base--uncased-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Dataset](https://img.shields.io/badge/RAID-Benchmark%20Corpus-3B82F6?style=for-the-badge)](https://github.com/liamdugan/raid)

An end-to-end deep learning system and web application designed to detect AI-generated text using a **length-invariant 1D Convolutional Neural Network (1D-CNN)** operating over **BERT contextual embeddings** and **auxiliary statistical NLP feature vectors**.

Trained and evaluated on the **RAID Benchmark Dataset** (5.6 Million generations across 11 LLM models), DeepTrace AI combines deep learning feature extraction with an interactive React workbench featuring real-time sliding window visualizations and sentence-level activation heatmaps.

---

## 📸 Interactive Workbench Demo

- **Detector Workbench**: Instant document scoring, confidence metrics, and interactive sentence-level risk heatmaps anchored directly to Conv1D sliding window activations.
- **Sliding Window Visualizer**: Step-by-step CSS/DOM interactive animation of 1D convolution filters ($k=1, 3, 5, 7$) sliding across BERT token embedding tracks.
- **RAID Analytics**: Comprehensive benchmark breakdown across generator models (GPT-4, Claude 3, LLaMA 3, Mistral) and topical domains.

---

## 🧠 System Architecture

DeepTrace AI employs a **hybrid neural-statistical feature fusion architecture** that combines Transformer latent spaces with multi-kernel spatial convolutions and length-invariant corpus metrics.

```
Input Text (T)
   │
   ├─► BERT Base Encoder (Frozen) ──► Token Embeddings [1 × L × 768]
   │                                  Attention Mask [1 × L]
   │                                         │
   │                                         ▼
   │                                 Parallel 1D Conv Layers
   │                                 • k=1 (Pointwise Token Identity)
   │                                 • k=3 (Trigram Phrase Patterns)
   │                                 • k=5 (Pentagram Clause Syntax)
   │                                 • k=7 (Septagram Sentence Rhythm)
   │                                         │
   │                                         ▼
   │                                 LayerNorm per Branch
   │                                 Attention-Masked Max Pooling over Time
   │                                         │
   │                                         ▼
   │                                 512-dim CNN Feature Map
   │                                         │
   ├─► Auxiliary Feature Extractor ──────────┴─► [6 Advanced Statistical NLP Metrics]
   │    1. Shannon Lexical Entropy                      │
   │    2. Repetition Index (Bigram/Trigram)            ▼
   │    3. Syntactic Clause Complexity Variance   Concat Fusion Layer [1286-dim]
   │    4. Root-TTR Vocabulary Rarity                   │
   │    5. Consecutive Sentence Cosine Drift            ▼
   │    6. Subword Fragmentation Density          Dense Classifier Head with LayerNorm
   │                                                    │
   └────────────────────────────────────────────────────┴─► Sigmoid ──► P_AI Score
```

### Key Technical Innovations

1. **Multi-Kernel 1D Convolutions ($k=1, 3, 5, 7$)**: Parallel 128-filter 1D convolutions capture structural stylistic patterns across multiple n-gram granularities simultaneously.
2. **Attention-Masked Pooling over Time**: Prevents padding tokens (`[PAD]`) from skewing short text representations by applying a $-10^9$ mask prior to `F.max_pool1d`.
3. **LayerNorm Channel Scaling & Zero-Centered Init**: Normalizes un-scaled 768-dim BERT `[CLS]` vectors and CNN feature maps to eliminate magnitude dominance before the classifier head.
4. **Coherent Conv1D Sentence Heatmaps**: Sentence risk scores are derived directly from token activation intensity using `detector_model.get_sliding_window_scores(embeddings, mask)`—requiring **0 extra forward passes** while guaranteeing 100% coherence with overall document probability.

---

## 📊 Benchmark & Evaluation Results

Evaluated on clean baseline samples from the **RAID Dataset** using a **Split-Before-Extract 4-Partition Split** to guarantee zero data leakage:

| Partition | Share | Purpose | Result |
| :--- | :--- | :--- | :--- |
| **Train Set** | 70% | PyTorch gradient descent & weight optimization | Loss: `0.1840` |
| **Validation Set** | 10% | Early stopping & learning rate scheduling | Val Acc: **85.60%** |
| **Calibration Set** | 10% | Threshold calibration ($\tau \in [0.35 \dots 0.65]$) | Calib Acc: **86.20%** ($\tau_{\text{opt}} = 0.3500$) |
| **Holdout Test Set**| 10% | Unseen benchmark evaluation | **Test Acc: 88.13%** \| **F1: 0.8834** \| **ROC-AUC: 0.9459** |

---

## 🔍 Engineering Retrospective & Architecture Trade-offs

A realistic review of the theoretical boundaries of 1D-CNN + Frozen BERT architectures:

- **Formality Bias**: Human formal writing (academic, legal, historical) naturally shares stylistic traits with LLM summary outputs (low typos, balanced syntax, clean transitions).
- **Surface N-Gram Exploitation**: While 1D Convolutions ($k \le 7$) excel at detecting surface phrase regularities, modern frontier models (Claude 3.5 Sonnet, GPT-4o) generate locally natural $n$-grams when prompted conversationally.
- **Latency vs. Accuracy Trade-off**: 1D-CNN + BERT feature extraction runs in **~15ms on CPU**, offering massive throughput advantages over multi-second zero-shot log-likelihood detectors (e.g. Binoculars or DetectGPT).

---

## 🛠️ Tech Stack & Directory Structure

- **Deep Learning & NLP**: PyTorch, HuggingFace Transformers (`bert-base-uncased`), NumPy, Scikit-learn
- **Backend API**: FastAPI, Uvicorn, Pydantic
- **Frontend Workbench**: React 18, Vite, Custom Modern Slate/Blue CSS Design System

```
my-app/
├── python/
│   ├── api.py               # FastAPI backend with dynamic config loading & zero-fallback rules
│   ├── model.py             # Length-invariant TextCNN1D with Attention-Masked Max Pooling
│   ├── feature_extractor.py # BERT encoder + 6-metric auxiliary statistical feature extractor
│   └── train.py             # 4-partition dataset loader, gradient clipping, & threshold tuner
├── src/
│   ├── components/
│   │   ├── DetectorWorkbench.jsx   # Interactive text analysis & sentence heatmap UI
│   │   ├── ArchitectureInspector.jsx# Interactive sliding window convolution visualizer
│   │   └── RaidBenchmark.jsx       # RAID dataset performance analytics tab
│   ├── App.jsx              # Main React application shell
│   └── index.css            # Custom enterprise AI detector design system
├── checkpoints/
│   ├── cnn_bert_detector.pth# Trained PyTorch model weights (state dict)
│   └── metrics.json         # Exported hyperparameters, optimal threshold & benchmark scores
├── .gitignore
├── package.json
└── README.md
```

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.9+
- Node.js 18+

### 2. Backend Setup
```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Install requirements
pip install torch transformers fastapi uvicorn datasets pandas numpy scikit-learn

# Run FastAPI Web Server
python python/api.py
```
The API server will start at `http://localhost:8000`.

### 3. Frontend Setup
```bash
# Install Node dependencies
npm install

# Start Vite dev server
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
