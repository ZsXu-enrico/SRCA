"""Training script for SRCA model on SEHGN dataset.

This script:
1. Builds category co-occurrence graph
2. Initializes SRCA model with TinyLlama
3. Trains with Focal Loss
4. Evaluates with P@K, R@K, NDCG@K, MAP@K
"""

import os
import sys
import time
from datetime import timedelta

# Set CUDA device to use 1 GPU (避免DDP内存重复)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from config import SRCA_CONFIG
from src.datamodules.srca_datamodule import SRCADataModule
from src.models.srca_model import SRCALightningModule
from src.utils.category_graph import MashupAPIGraph

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_gpu_memory(stage=""):
    """Print GPU memory usage."""
    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
        reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
        max_allocated_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        max_allocated_gb = max_allocated_mb / 1024

        print(f"\n{'='*70}")
        print(f"GPU Memory Usage {stage}")
        print(f"{'='*70}")
        print(f"  Allocated:     {allocated_mb:>10.2f} MB ({allocated_mb/1024:>6.2f} GB)")
        print(f"  Reserved:      {reserved_mb:>10.2f} MB ({reserved_mb/1024:>6.2f} GB)")
        print(f"  Max Allocated: {max_allocated_mb:>10.2f} MB ({max_allocated_gb:>6.2f} GB)")
        print(f"{'='*70}\n")


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("SRCA Model Training with TinyLlama on SEHGN Dataset")
    print("=" * 70)

    print("\nConfiguration:")
    for key, value in SRCA_CONFIG.items():
        print(f"  {key}: {value}")

    # Set seed
    pl.seed_everything(SRCA_CONFIG['seed'], workers=True)

    # Enable Tensor Cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("\n⚠ No GPU available, using CPU")

    # ============================================
    # Step 1: Initialize DataModule
    # ============================================
    print("\n" + "=" * 70)
    print("Step 1: Loading Data")
    print("=" * 70)

    datamodule = SRCADataModule(
        data_dir=SRCA_CONFIG['data_dir'],
        batch_size=SRCA_CONFIG['batch_size'],
        num_workers=SRCA_CONFIG['num_workers']
    )

    datamodule.prepare_data()
    datamodule.setup()

    print(f"✓ Train: {len(datamodule.train_dataset)} samples")
    print(f"✓ Val: {len(datamodule.val_dataset)} samples")
    print(f"✓ Test: {len(datamodule.test_dataset)} samples")

    # ============================================
    # Step 2: Build Mashup-API Category Co-occurrence Graph
    # ============================================
    print("\n" + "=" * 70)
    print("Step 2: Building Mashup-API Category Co-occurrence Graph")
    print("=" * 70)

    # Collect mashup categories from train + val + test
    # Transductive setting: graph includes all mashups (paper Section 5.1)
    # DataModule loads all 8217 mashups, uses 80/10/10 split for train/val/test
    # Get all mashup categories (all 8217 from datamodule)
    mashup_categories = datamodule.mashup_categories

    # Load API categories from JSON files (same as datamodule)
    import pandas as pd
    import json

    # Use source_data_dir from config (points to /home/zxu298/TRELLIS/api/data/)
    source_data_dir = SRCA_CONFIG['source_data_dir']
    api_cat_file = os.path.join(source_data_dir, 'api_category.json')
    api_name_file = os.path.join(source_data_dir, 'api_name.json')
    used_api_file = os.path.join(source_data_dir, 'used_api_list.json')

    # Load all API names and categories (23518 total)
    with open(api_name_file, 'r') as f:
        all_api_names = json.load(f)
    with open(api_cat_file, 'r') as f:
        all_api_categories = json.load(f)
    with open(used_api_file, 'r') as f:
        used_api_names = json.load(f)

    # Build name-to-index mapping
    api_name_to_idx = {name: idx for idx, name in enumerate(all_api_names)}

    # Extract categories for used APIs only (1647 APIs)
    api_categories = []
    for used_name in used_api_names:
        if used_name in api_name_to_idx:
            idx = api_name_to_idx[used_name]
            api_categories.append(all_api_categories[idx])
        else:
            api_categories.append([])

    print(f"Collected {len(mashup_categories)} mashups and {len(api_categories)} APIs")

    # Build Mashup-API bipartite graph
    graph_builder = MashupAPIGraph(
        threshold=SRCA_CONFIG['edge_threshold']
    )
    edge_index, edge_weight = graph_builder.build_from_data(
        mashup_categories,
        api_categories
    )

    print(f"✓ Mashup-API graph built:")
    print(f"  - {graph_builder.num_mashups} mashups")
    print(f"  - {graph_builder.num_apis} APIs")
    print(f"  - {edge_index.size(1)} edges")

    # Save graph
    graph_path = os.path.join(SRCA_CONFIG['data_dir'], 'category_graph.pt')
    graph_builder.save(graph_path)
    print(f"✓ Saved category graph to {graph_path}")

    # ============================================
    # Step 3: Load or Extract Semantic Features
    # ============================================
    print("\n" + "=" * 70)
    print("Step 3: Loading Semantic Features")
    print("=" * 70)

    feature_file = os.path.join(SRCA_CONFIG['data_dir'], 'semantic_features.pt')

    if not os.path.exists(feature_file):
        print(f"\n⚠ Features not found at {feature_file}")
        print("Please run feature extraction first:")
        print("  conda run -n aaai python extract_features.py")
        print("\nExtracting features now...")

        # Run feature extraction
        import subprocess
        result = subprocess.run(
            ["conda", "run", "-n", "aaai", "python", "extract_features.py"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"❌ Feature extraction failed:")
            print(result.stderr)
            sys.exit(1)

        print("✓ Feature extraction completed")

    # Load features
    print(f"Loading features from {feature_file}...")
    features_data = torch.load(feature_file, map_location='cpu', weights_only=False)

    # Use features for all mashups (8217 in our dataset, 7739 in paper)
    num_mashups_used = datamodule.num_mashups  # All 8217 mashups
    mashup_features = features_data['mashup_features'][:num_mashups_used]
    api_features = features_data['api_features']

    print(f"✓ Loaded semantic features:")
    print(f"  - Mashup features: {mashup_features.shape} (using all {num_mashups_used})")
    print(f"  - API features: {api_features.shape}")

    # ============================================
    # Step 4: Initialize SRCA Model
    # ============================================
    print("\n" + "=" * 70)
    print("Step 4: Initializing SRCA Model")
    print("=" * 70)

    model = SRCALightningModule(
        num_apis=SRCA_CONFIG['num_apis'],
        num_mashups=datamodule.num_mashups,  # Use actual num_mashups (6574, not 8217)
        semantic_dim=SRCA_CONFIG['semantic_dim'],
        gnn_num_layers=SRCA_CONFIG['gnn_num_layers'],
        mlp_hidden_dims=SRCA_CONFIG['mlp_hidden_dims'],
        mlp_dropout=SRCA_CONFIG['mlp_dropout'],
        mashup_features=mashup_features,
        api_features=api_features,
        focal_alpha=SRCA_CONFIG['focal_alpha'],
        focal_gamma=SRCA_CONFIG['focal_gamma'],
        learning_rate=SRCA_CONFIG['learning_rate'],
        weight_decay=SRCA_CONFIG['weight_decay'],
        mashup_api_graph_edge_index=edge_index,
        mashup_api_graph_edge_weight=edge_weight,
        eval_k_values=[1, 3, 5, 10]
    )

    print("✓ SRCA Model initialized")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")


    # ============================================
    # Step 4.5: Pre-compute GNN-augmented features
    # ============================================
    print("\n" + "=" * 70)
    print("Step 4.5: Pre-computing GNN-Augmented Features")
    print("=" * 70)
    print("Computing augmented features once to save memory...")

    # Move model to GPU before computing
    if torch.cuda.is_available():
        model = model.cuda()

    # Pre-compute GNN features (done once, saves memory during training)
    model.compute_augmented_features()

    print("✓ GNN features pre-computed and cached")
    print("  Only MLP will be trained (GNN features are frozen)")

    # Free up memory
    torch.cuda.empty_cache()

    # ============================================
    # Step 5: Setup Training
    # ============================================
    print("\n" + "=" * 70)
    print("Step 5: Setting up Training")
    print("=" * 70)

    # Callbacks - save best model based on validation MAP@5 (paper's main metric)
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='srca-best-{epoch:02d}-{val/MAP@5:.4f}',
        monitor='val/MAP@5',  # Paper reports MAP@K as main metric
        mode='max',
        save_top_k=1,  # Save best model
        save_last=True,  # Also save last checkpoint
        verbose=True
    )

    # Logger
    logger_tb = TensorBoardLogger(
        save_dir='./logs',
        name='srca_tinyllama'
    )

    # Trainer - Using 1 GPU to avoid DDP memory overhead
    trainer = pl.Trainer(
        max_epochs=SRCA_CONFIG['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,  # 单GPU训练避免内存重复
        callbacks=[checkpoint_callback],
        logger=logger_tb,
        log_every_n_steps=10,
        deterministic=True,
        precision='16-mixed' if torch.cuda.is_available() else '32',
        val_check_interval=1.0,  # Validate every epoch (paper uses 80/10/10 split)
    )

    print("✓ Trainer configured")
    print(f"  Max epochs: {SRCA_CONFIG['max_epochs']}")
    print(f"  GPU: 1 (L20 48GB)")
    print(f"  Precision: {'16-mixed' if torch.cuda.is_available() else '32'}")

    # ============================================
    # Step 6: Train Model
    # ============================================
    print("\n" + "=" * 70)
    print("Step 6: Training SRCA Model")
    print("=" * 70)

    try:
        # Start training timer
        train_start_time = time.time()

        trainer.fit(model, datamodule=datamodule)

        # Calculate training time
        train_end_time = time.time()
        train_duration = train_end_time - train_start_time

        print("\n" + "=" * 70)
        print("Training Completed!")
        print("=" * 70)
        print(f"Training time: {str(timedelta(seconds=int(train_duration)))}")
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Best val/MAP@5: {checkpoint_callback.best_model_score:.4f}")

        # ============================================
        # Step 7: Test Model
        # ============================================
        print("\n" + "=" * 70)
        print("Step 7: Testing SRCA Model")
        print("=" * 70)

        # Reset GPU memory stats before testing
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        print_gpu_memory("(Before Testing)")

        # Start testing timer
        test_start_time = time.time()

        # Use the best checkpoint based on validation MAP@5
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path='best'  # Use best model from validation
        )

        # Calculate testing time
        test_end_time = time.time()
        test_duration = test_end_time - test_start_time

        # Print GPU memory after testing
        print_gpu_memory("(After Testing)")

        print("\n" + "=" * 70)
        print("SRCA Training and Evaluation Completed Successfully!")
        print("=" * 70)
        print(f"Training time: {str(timedelta(seconds=int(train_duration)))}")
        print(f"Testing time: {str(timedelta(seconds=int(test_duration)))}")
        print(f"Total time: {str(timedelta(seconds=int(train_duration + test_duration)))}")
        print("\nTo view training logs, run:")
        print("  tensorboard --logdir=./logs")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
