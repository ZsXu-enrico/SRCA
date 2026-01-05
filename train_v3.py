"""Training script for SRCA V3 - 论文正确实现.

关键改进：
1. Mashup-API服务图（基于类别Jaccard）
2. 只用训练集构建图（避免数据泄露）
3. Token-level features + learnable aggregation
"""

import os
import sys
import time
from datetime import timedelta

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, os.path.dirname(__file__))

from config import SRCA_CONFIG
from src.datamodules.srca_datamodule import SRCADataModule
from src.models.srca_model_v3 import SRCALightningModuleV3
from src.utils.mashup_api_category_graph import MashupAPICategoryGraph

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline for SRCA V3."""
    print("=" * 70)
    print("SRCA V3: 论文正确实现")
    print("=" * 70)

    print("\nConfiguration:")
    for key, value in SRCA_CONFIG.items():
        print(f"  {key}: {value}")

    # Set seed
    pl.seed_everything(SRCA_CONFIG['seed'], workers=True)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")

    # Step 1: Initialize DataModule
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

    # Step 2: Build Mashup-API graph (只用训练集)
    print("\n" + "=" * 70)
    print("Step 2: Building Mashup-API Graph (Training Set Only)")
    print("=" * 70)

    # 获取训练集mashup的类别
    train_mashup_categories = datamodule.mashup_categories[:len(datamodule.train_dataset)]
    api_categories = datamodule.api_categories

    print(f"Using training data:")
    print(f"  Train mashups: {len(train_mashup_categories)}")
    print(f"  All APIs: {len(api_categories)}")

    # 构建图
    graph_path = os.path.join(SRCA_CONFIG['data_dir'], 'mashup_api_graph_train_only.pt')

    if os.path.exists(graph_path):
        print(f"Loading existing graph from {graph_path}...")
        graph_builder = MashupAPICategoryGraph(threshold=SRCA_CONFIG['cooccur_threshold'])
        graph_builder.load(graph_path)
        edge_index = graph_builder.edge_index
        edge_weight = graph_builder.edge_weight
    else:
        print("Building new graph...")
        graph_builder = MashupAPICategoryGraph(threshold=SRCA_CONFIG['cooccur_threshold'])
        edge_index, edge_weight = graph_builder.build_from_train_data(
            train_mashup_categories,
            api_categories
        )
        graph_builder.save(graph_path)

    print(f"✓ Graph built: {edge_index.size(1)} edges")
    print(f"  Nodes: {graph_builder.num_train_mashups} train mashups + {graph_builder.num_apis} APIs")

    # Step 3: Load token features
    print("\n" + "=" * 70)
    print("Step 3: Loading Token-Level Features")
    print("=" * 70)

    token_file = os.path.join(SRCA_CONFIG['data_dir'], 'token_features.pt')
    if not os.path.exists(token_file):
        print(f"\n⚠ Token features not found at {token_file}")
        print("Please run: python extract_features_token_level.py")
        sys.exit(1)

    print(f"Loading from {token_file}...")
    token_data = torch.load(token_file, map_location='cpu', weights_only=False)

    # 只用训练集的mashup token features
    num_train_mashups = len(datamodule.train_dataset)
    train_mashup_token_features = token_data['mashup_token_features'][:num_train_mashups]
    api_token_features = token_data['api_token_features']
    seq_len = token_data['max_tokens']

    print(f"✓ Loaded token features:")
    print(f"  - Train Mashup: {train_mashup_token_features.shape}")
    print(f"  - API: {api_token_features.shape}")
    print(f"  - Sequence length: {seq_len}")

    # Step 4: Initialize Model
    print("\n" + "=" * 70)
    print("Step 4: Initializing SRCA V3 Model")
    print("=" * 70)

    model = SRCALightningModuleV3(
        num_apis=SRCA_CONFIG['num_apis'],
        num_train_mashups=num_train_mashups,
        seq_len=seq_len,
        semantic_dim=SRCA_CONFIG['semantic_dim'],
        gnn_num_layers=SRCA_CONFIG['gnn_num_layers'],
        mlp_hidden_dims=SRCA_CONFIG['mlp_hidden_dims'],
        mlp_dropout=SRCA_CONFIG['mlp_dropout'],
        aggregation_type='attention',
        train_mashup_token_features=train_mashup_token_features,
        api_token_features=api_token_features,
        focal_alpha=SRCA_CONFIG['focal_alpha'],
        focal_gamma=SRCA_CONFIG['focal_gamma'],
        learning_rate=SRCA_CONFIG['learning_rate'],
        weight_decay=SRCA_CONFIG['weight_decay'],
        mashup_api_graph_edge_index=edge_index,
        mashup_api_graph_edge_weight=edge_weight,
        eval_k_values=[1, 3, 5, 10]
    )

    print("✓ SRCA V3 Model initialized")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Step 5: Setup Training
    print("\n" + "=" * 70)
    print("Step 5: Setting up Training")
    print("=" * 70)

    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='srca-v3-last',
        save_top_k=0,
        save_last=True,
        verbose=False
    )

    logger_tb = TensorBoardLogger(
        save_dir='./logs',
        name='srca_v3'
    )

    trainer = pl.Trainer(
        max_epochs=SRCA_CONFIG['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=logger_tb,
        log_every_n_steps=10,
        deterministic=True,
        precision='16-mixed' if torch.cuda.is_available() else '32',
        limit_val_batches=0,
    )

    print("✓ Trainer configured")
    print(f"  Max epochs: {SRCA_CONFIG['max_epochs']}")

    # Step 6: Train
    print("\n" + "=" * 70)
    print("Step 6: Training SRCA V3")
    print("=" * 70)

    try:
        train_start_time = time.time()
        trainer.fit(model, datamodule=datamodule)
        train_end_time = time.time()
        train_duration = train_end_time - train_start_time

        print("\n" + "=" * 70)
        print("Training Completed!")
        print("=" * 70)
        print(f"Training time: {str(timedelta(seconds=int(train_duration)))}")
        print(f"Last checkpoint: {checkpoint_callback.last_model_path}")

        # Step 7: Test
        print("\n" + "=" * 70)
        print("Step 7: Testing SRCA V3")
        print("=" * 70)

        test_start_time = time.time()
        trainer.test(model=model, datamodule=datamodule, ckpt_path='last')
        test_end_time = time.time()
        test_duration = test_end_time - test_start_time

        print("\n" + "=" * 70)
        print("SRCA V3 Training and Evaluation Completed!")
        print("=" * 70)
        print(f"Training time: {str(timedelta(seconds=int(train_duration)))}")
        print(f"Testing time: {str(timedelta(seconds=int(test_duration)))}")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
