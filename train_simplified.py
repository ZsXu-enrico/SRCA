"""Simplified Training script for SRCA model (without GNN for now).

This simplified version:
1. Extracts LLM semantic features
2. Trains recommendation MLP
3. Skips GNN augmentation (to be added later)

TODO: Add full Mashup-API graph construction and GNN augmentation
"""

import os
import sys

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, os.path.dirname(__file__))

from config import SRCA_CONFIG
from src.datamodules.srca_datamodule import SRCADataModule
from src.models.srca_model import SRCALightningModule

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("SRCA Model Training (Simplified - No GNN)")
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
    # Step 2: Initialize SRCA Model (No GNN)
    # ============================================
    print("\n" + "=" * 70)
    print("Step 2: Initializing SRCA Model")
    print("=" * 70)

    model = SRCALightningModule(
        num_apis=SRCA_CONFIG['num_apis'],
        semantic_dim=SRCA_CONFIG['semantic_dim'],
        gnn_num_layers=SRCA_CONFIG['gnn_num_layers'],
        mlp_hidden_dims=SRCA_CONFIG['mlp_hidden_dims'],
        mlp_dropout=SRCA_CONFIG['mlp_dropout'],
        llm_model_name=SRCA_CONFIG['llm_model_name'],
        llm_max_length=SRCA_CONFIG['llm_max_length'],
        freeze_llm=True,
        focal_alpha=SRCA_CONFIG['focal_alpha'],
        focal_gamma=SRCA_CONFIG['focal_gamma'],
        learning_rate=SRCA_CONFIG['learning_rate'],
        weight_decay=SRCA_CONFIG['weight_decay'],
        # No graph for now
        mashup_api_graph_edge_index=None,
        mashup_api_graph_edge_weight=None,
        eval_k_values=[1, 3, 5, 10]
    )

    print("✓ SRCA Model initialized (without GNN augmentation)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ============================================
    # Step 3: Setup Training
    # ============================================
    print("\n" + "=" * 70)
    print("Step 3: Setting up Training")
    print("=" * 70)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='srca-{epoch:02d}-{val/P@5:.4f}',
        monitor='val/P@5',
        mode='max',
        save_top_k=3,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val/P@5',
        patience=SRCA_CONFIG['patience'],
        mode='max',
        verbose=True
    )

    # Logger
    logger_tb = TensorBoardLogger('lightning_logs', name='srca')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=SRCA_CONFIG['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger_tb,
        deterministic=True,
        precision=16  # Use mixed precision for speed
    )

    print("✓ Trainer configured")

    # ============================================
    # Step 4: Train
    # ============================================
    print("\n" + "=" * 70)
    print("Step 4: Training")
    print("=" * 70)

    try:
        trainer.fit(model, datamodule=datamodule)
        print("\n✓ Training completed successfully!")

        # Test
        print("\n" + "=" * 70)
        print("Testing")
        print("=" * 70)
        trainer.test(model, datamodule=datamodule)

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
