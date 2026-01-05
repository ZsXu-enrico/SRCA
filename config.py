"""
Configuration for SRCA model - Based on paper implementation.

Paper: LLM-enhanced service Semantic Representation and Category
       co-occurrence feature Augmentation for Web API recommendation

Dataset: ProgrammableWeb
- Mashups: 8217
- APIs (used): 1647
- Categories: 499
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SRCA_CONFIG = {
    # ===== LLM Configuration (Section 4.1) =====
    # Paper uses LLaMA-3 series (1B, 3B, 8B)
    # Best result with LLaMA-3.1-8B-Instruct
    'llm_model_path': os.path.join(BASE_DIR, '../Llama3.1-8B-hf'),  # LLaMA-3.1-8B local path
    'llm_max_length': 512,  # Max tokens for input
    'llm_hidden_dim': 4096,  # LLaMA-3-8B hidden dimension
    'use_4bit': False,  # Set True if memory limited
    'use_8bit': False,

    # ===== Semantic Features (Section 4.1.2) =====
    # Paper: Extract from final layer, mean pooling, then dimension mapping
    'semantic_dim': 1024,  # Mapped dimension (Paper uses 1024)

    # ===== Category Co-occurrence Graph (Section 4.2.1) =====
    # Paper uses Jaccard coefficient with threshold θ = 0.2
    'edge_threshold': 0.2,  # θ threshold for edge formation

    # ===== GNN Feature Augmentation (Section 4.2.2) =====
    # Paper: 2-layer GNN with LayerNorm, ELU, residual connections
    'gnn_num_layers': 2,  # K = 2 layers
    'gnn_hidden_dim': 1024,  # Same as semantic_dim

    # ===== Recommendation MLP (Section 4.3) =====
    # Paper: 2-layer MLP with hidden dim 512
    'mlp_hidden_dims': [512],  # One hidden layer of 512 dims
    'mlp_dropout': 0.3,

    # ===== Focal Loss (Section 4.4) =====
    # Paper parameters for handling class imbalance
    'focal_alpha': 0.25,  # α balancing factor
    'focal_gamma': 2.0,   # γ focusing parameter

    # ===== Training (Section 5.4) =====
    # Paper: Adam optimizer with specific hyperparameters
    'batch_size': 64,  # Paper uses 64
    'learning_rate': 1e-4,  # Paper uses 1e-4
    'weight_decay': 0.0005,   # 5e-4 (L2 regularization λ)
    'max_epochs': 100,
    'patience': 15,  # Early stopping patience

    # ===== Data =====
    # ProgrammableWeb dataset - YOUR ACTUAL DATA
    'data_dir': os.path.join(BASE_DIR, 'data'),
    'source_data_dir': os.path.join(BASE_DIR, '../data'),  # Original JSON files
    'num_apis': 1647,      # YOUR DATA: 1647 used APIs
    'num_mashups': 8217,   # YOUR DATA: 8217 mashups
    'num_categories': 499,  # YOUR DATA: 499 categories
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,

    # ===== Evaluation (Section 5.2) =====
    # Paper evaluates at N = {5, 10, 15, 20}
    'eval_k_values': [5, 10, 15, 20],

    # ===== System =====
    'seed': 42,
    'num_workers': 4,
    'device': 'cuda',
}
