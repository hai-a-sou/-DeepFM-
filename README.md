# DeepFM for CTR Prediction on Criteo Dataset

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of **DeepFM** (Deep Factorization Machine) for click-through rate (CTR) prediction on the Criteo display advertising dataset. Includes FM and XGBoost baselines for comparison, with a streaming preprocessing pipeline that handles the full 45-million-row dataset without loading it into memory.

## Project Overview

CTR prediction is a core task in computational advertising and recommender systems. This project explores how different orders of feature interaction affect predictive performance:

- **FM** captures pairwise (2nd-order) feature interactions via latent vectors
- **DeepFM** adds a DNN on top of shared embeddings to learn higher-order interactions
- **XGBoost** serves as a tree-based baseline that learns feature splits

The project uses the **Kaggle Criteo dataset** — 45 million display ad impressions with 13 numerical and 26 categorical features — and achieves **validation AUC ≥ 0.81** with DeepFM.

## Model Architecture

```
Input: 13 Numerical + 26 Categorical Features
                |
    ┌───────────┴───────────┐
    |         Shared         |
    |    Embedding Layer     |
    |   (39 fields × d)     |
    └───┬───────┬───────┬───┘
        |       |       |
   ┌────v────┐ ┌v───┐ ┌─v──────────┐
   │ Linear  │ │ FM │ │    DNN      │
   │(1st-    │ │(2nd│ │ (high-order)│
   │ order)  │ │order)│ │[512→256→128]
   └───┬─────┘ └─┬──┘ └────┬───────┘
       └──────────┼─────────┘
                  v
            Sum + Sigmoid
                  |
               Output
         P(click | features)
```

**Key components:**

| Component | Description |
|-----------|-------------|
| **FeaturesLinear** | First-order weights: `bias + Σwᵢxᵢ`. Categorical features use EmbeddingBag, numerical use learned scalar weights |
| **FeaturesEmbedding** | Per-field embedding lookup shared by FM and DNN. Numerical fields use a learned vector scaled by feature value |
| **FactorizationMachine** | O(nk) pairwise interaction: `0.5·Σₖ((Σᵢvᵢₖ)² - Σᵢvᵢₖ²)` |
| **MLP** | 3-layer fully-connected network (512→256→128→1) with BatchNorm, ReLU, Dropout(0.2) |

## Results

| Model | Test AUC | Test LogLoss | Parameters |
|-------|----------|-------------|------------|
| XGBoost | ~0.790 | ~0.470 | — |
| FM (embed_dim=16) | ~0.798 | ~0.465 | ~10M |
| **DeepFM (embed_dim=16)** | **~0.808** | **~0.456** | ~12M |
| DeepFM (embed_dim=24) | ~0.812 | ~0.452 | ~14M |

ROC curves and training curves are generated in [`results/figures/`](results/figures/).

## Dataset

The [Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge) dataset:

- **45 million** click records (split into train.txt, ~11 GB)
- **13 numerical** features (I1–I13, may contain missing values)
- **26 categorical** features (C1–C26, hex-encoded hashes)
- **Label**: 1 = click, 0 = not click

**Option A — HuggingFace (recommended, no auth required):**

```bash
pip install datasets tqdm
python data/download_from_hf.py
```

**Option B — Kaggle (requires Kaggle account):**

```bash
pip install kaggle
kaggle competitions download -c criteo-display-ad-challenge
unzip criteo-display-ad-challenge.zip -d data/
```

## Preprocessing Pipeline

The preprocessing handles the 45M-row dataset in **two streaming passes** (never loads full file into memory):

```
Raw TSV (13 GB)
    |
    ├── Pass 1 (fit): read_csv(chunksize=500K)
    |   ├── Numerical: Welford online mean/std
    |   └── Categorical: frequency counting
    |
    ├── Pass 2 (transform): re-stream + apply
    |   ├── Numerical: log1p → standardize
    |   ├── Categorical: map to int indices (freq < 10 → <UNK>)
    |   └── Output: numpy .npy memmap files (~7 GB)
    |
    └── Split: 80% train / 10% val / 10% test
```

Key design choices:
- **Log transform** on numerical features (highly skewed distributions)
- **Frequency-based bucketing**: values appearing < 10 times mapped to `<UNK>` token
- **Max vocabulary cap**: 500K per categorical field to bound embedding size
- **NumPy memmap**: OS-level paging enables random access to any row without loading into RAM

## Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml              # All hyperparameters
│
├── deepfm_ctr/                   # Main package
│   ├── config.py                 # YAML config loader
│   ├── data/
│   │   ├── preprocessing.py      # CriteoPreprocessor (streaming 2-pass)
│   │   └── dataset.py            # CriteoDataset + DataLoaders
│   ├── models/
│   │   ├── layers.py             # FeaturesLinear, Embedding, FM, MLP
│   │   ├── fm.py                 # FactorizationMachineModel
│   │   └── deepfm.py             # DeepFM
│   ├── training/
│   │   ├── trainer.py            # Early stopping, checkpointing
│   │   └── evaluator.py          # AUC, LogLoss, metrics
│   ├── utils/
│   │   ├── logger.py
│   │   └── reproducibility.py    # seed_everything, get_device
│   └── visualization/
│       └── plotter.py            # ROC, training curves, comparisons
│
├── experiments/
│   ├── 01_preprocess.py          # Run preprocessing
│   ├── 02_train_fm.py            # FM baseline
│   ├── 03_train_deepfm.py        # DeepFM main model
│   ├── 04_train_xgboost.py       # XGBoost baseline
│   ├── 05_search_embedding_dim.py # Embedding dim grid search
│   └── 06_final_comparison.py    # Aggregate results
│
├── data/                         # Raw + processed data (gitignored)
└── results/                      # Logs, checkpoints, figures (gitignored)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and preprocess data

```bash
# Option A: HF-Mirror (recommended, no auth)
pip install datasets tqdm
python data/download_from_hf.py

# Option B: Kaggle (requires Kaggle account)
# Download from Kaggle and place train.txt in data/

# Run preprocessing (~30-60 min, two passes over the dataset)
python experiments/01_preprocess.py
```

### 3. Train models

```bash
# Train FM baseline (~2-3 hours on GPU)
python experiments/02_train_fm.py

# Train DeepFM (~3-4 hours on GPU)
python experiments/03_train_deepfm.py

# Train XGBoost baseline (~1-2 hours on CPU)
python experiments/04_train_xgboost.py

# Search embedding dimensions (~8-12 hours total)
python experiments/05_search_embedding_dim.py

# Generate comparison figures
python experiments/06_final_comparison.py
```

### 4. View results

All metrics, figures, and checkpoints are saved in the `results/` directory:

- `results/figures/roc_comparison.png` — ROC curves for all models
- `results/figures/embedding_dim_search.png` — AUC vs embedding dimension
- `results/figures/model_comparison.png` — AUC and LogLoss bar charts
- `results/*_metrics.json` — Full evaluation metrics

## Configuration

All hyperparameters are in [`configs/default.yaml`](configs/default.yaml):

```yaml
model:
  embed_dim: 16          # Embedding dimension
  mlp_dims: [512, 256, 128]  # MLP hidden layer sizes
  dropout: 0.2
  use_batch_norm: true

training:
  batch_size: 4096
  learning_rate: 0.001
  early_stopping_patience: 3

search:
  embed_dims: [4, 8, 12, 16, 24, 32]  # Grid for embedding search
```

## Key Findings

1. **Embedding dimension matters**: AUC increases from 0.80 at dim=4 to 0.81+ at dim=16, with diminishing returns beyond dim=24
2. **DeepFM > FM > XGBoost**: The deep component adds ~0.01 AUC over pure FM by learning high-order interactions that pairwise factorization misses
3. **Sparse categorical features dominate**: 26 categorical fields account for most of the model parameters, and their embedding quality is critical
4. **Early stopping is essential**: Without it, DeepFM overfits after ~10 epochs; with patience=3, training reliably converges near the optimal point

## References

- [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) — Guo et al., IJCAI 2017
- [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) — Rendle, ICDM 2010
- [Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge) — Kaggle 2014

## License

MIT License
