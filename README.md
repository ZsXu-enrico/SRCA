# SRCA Model Implementation with TinyLlama

Implementation of **SRCA (Semantic Representation with Category Augmentation)** for API recommendation using TinyLlama-1.1B instead of full LLaMA-3-8B for efficiency.

## Overview

SRCA enhances API recommendation by combining:
1. **LLM-based Semantic Representation**: Uses TinyLlama to generate unified descriptions via prompts (RPM for mashups, FPA for APIs)
2. **Category Co-occurrence Graph**: Captures relationships between service categories
3. **GNN-based Feature Augmentation**: Enriches semantic features with category graph information
4. **Focal Loss**: Addresses class imbalance in API recommendation

## Architecture

```
Mashup Description
       ↓
[TinyLlama Encoder] ← RPM Prompt
       ↓
  Semantic Features (768-dim)
       ↓
[Category Features] → [GNN (2 layers)] → Augmented Features
       ↓                                          ↓
       └─────────── Feature Fusion ──────────────┘
                           ↓
                    [MLP + Focal Loss]
                           ↓
                   API Recommendations
```

## Key Components

### 1. LLM Semantic Encoder (`src/models/llm_semantic.py`)
- **Model**: TinyLlama-1.1B-Chat-v1.0
- **RPM Prompt**: Unifies mashup descriptions
- **FPA Prompt**: Extracts API features
- **Output**: 768-dimensional semantic features

### 2. Category Co-occurrence Graph (`src/utils/category_graph.py`)
- Builds graph from category co-occurrence in services
- Edges weighted by co-occurrence frequency
- Threshold: 0.01 (configurable)

### 3. GNN Feature Augmentation (`src/models/gnn_augmentation.py`)
- **Architecture**: 2-layer GCN
- **Hidden dim**: 256
- **Formula**: `h_final = θ * h_aug + (1-θ) * h_cat` where θ=0.2

### 4. Recommendation MLP (`src/models/recommendation_mlp.py`)
- **Architecture**: [768 → 512 → 256 → 128 → num_apis]
- **Dropout**: 0.3
- **Loss**: Focal Loss (α=0.25, γ=2.0)

## Installation

```bash
# Activate aaai environment (should have PyTorch 2.8.0 already)
conda activate aaai

# Install additional requirements
cd /home/zxu298/TRELLIS/api/AdaFlow/SRCA
pip install -r requirements.txt
```

## Dataset

Uses the same SEHGN dataset as AdaFlow:
- **Location**: `../data/SEHGN/`
- **Mashups**: 8,217
- **APIs**: 1,647
- **Split**: 60% train, 20% val, 20% test

## Usage

### Training

```bash
# Full training with TinyLlama
python train.py
```

### Configuration

Edit `config.py` to modify hyperparameters:

```python
SRCA_CONFIG = {
    # LLM
    'llm_model_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'semantic_dim': 768,

    # GNN
    'gnn_num_layers': 2,
    'gnn_hidden_dim': 256,
    'gnn_theta': 0.2,

    # Focal Loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,

    # Training
    'batch_size': 32,
    'learning_rate': 1e-4,
    'max_epochs': 100,
    'patience': 10,
}
```

## Testing Individual Components

```bash
# Test LLM encoder
python -c "from src.models.llm_semantic import test_llm_encoder; test_llm_encoder()"

# Test category graph
python -c "from src.utils.category_graph import test_category_graph; test_category_graph()"

# Test feature augmentation
python -c "from src.models.gnn_augmentation import test_feature_augmentation; test_feature_augmentation()"

# Test recommendation MLP
python -c "from src.models.recommendation_mlp import test_recommendation_mlp; test_recommendation_mlp()"

# Test focal loss
python -c "from src.models.focal_loss import test_focal_loss; test_focal_loss()"
```

## Evaluation Metrics

The model is evaluated on:
- **Precision@K**: Accuracy of top-K recommendations
- **Recall@K**: Coverage of relevant APIs in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision

for K ∈ {1, 3, 5, 10}

## Output

Training produces:
- `checkpoints/` - Best model checkpoints
- `logs/srca_tinyllama/` - TensorBoard logs
- `../data/SEHGN/category_graph.pt` - Category co-occurrence graph

View training progress:
```bash
tensorboard --logdir=./logs
```

## Differences from Original SRCA Paper

1. **LLM Model**: TinyLlama-1.1B instead of LLaMA-3-8B
   - Reason: Resource efficiency while maintaining semantic understanding
   - Hidden size: 2048 (projected to 768)

2. **Semantic Dim**: 768 instead of paper's dimension
   - Compatible with TinyLlama's architecture

3. **Training Setup**: Mixed precision (FP16) on GPU for efficiency

## Expected Performance

Based on SRCA paper (with full LLaMA-3-8B):
- **MAP@5**: 0.7421
- **NDCG@5**: 0.8031
- **P@5**: ~0.65

With TinyLlama, expect slightly lower but competitive performance (~3-5% drop).

## File Structure

```
SRCA/
├── config.py                          # Configuration
├── train.py                           # Main training script
├── requirements.txt                   # Dependencies
├── README.md                          # This file
└── src/
    ├── models/
    │   ├── llm_semantic.py           # TinyLlama encoder
    │   ├── gnn_augmentation.py       # GNN feature augmentation
    │   ├── recommendation_mlp.py     # MLP + Focal Loss
    │   ├── focal_loss.py             # Focal Loss implementation
    │   └── srca_model.py             # Complete SRCA model
    ├── datamodules/
    │   └── srca_datamodule.py        # Data loading
    └── utils/
        ├── prompts.py                 # RPM/FPA prompt templates
        └── category_graph.py          # Category graph construction
```

## Hardware Requirements

- **GPU**: NVIDIA RTX A6000 (48GB) or similar
- **RAM**: 32GB+ recommended
- **Disk**: ~20GB for model and data

## Troubleshooting

1. **Out of Memory**: Reduce batch_size in config.py
2. **Slow Training**: Ensure GPU is being used (`torch.cuda.is_available()`)
3. **Import Errors**: Verify all packages in requirements.txt are installed

## References

- SRCA Paper: "Semantic Representation with Category Augmentation for Web API Recommendation"
- TinyLlama: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)

## Citation

```bibtex
@article{srca2024,
  title={Semantic Representation with Category Augmentation for Web API Recommendation},
  author={...},
  journal={...},
  year={2024}
}
```

---

**Status**: ✅ Complete implementation ready for training
**Date**: 2025-10-08
