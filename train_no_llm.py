"""Training script for SRCA model without LLM semantic features.

This is an ablation study to test the contribution of:
- Learnable embeddings vs LLM semantic features
- GNN with category co-occurrence graph
- Focal Loss for class imbalance

Usage:
    python train_no_llm.py --batch_size 64 --max_epochs 100
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.srca_no_llm import SRCANoLLM
from src.utils.category_graph import MashupAPIGraph
from src.datamodules.srca_datamodule import SRCADataModule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train SRCA model without LLM')

    # Data
    parser.add_argument('--data_dir', type=str, default='data/ProgrammableWeb',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    # Model
    parser.add_argument('--embedding_dim', type=int, default=768,
                       help='Embedding dimension')
    parser.add_argument('--gnn_num_layers', type=int, default=2,
                       help='Number of GNN layers')
    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+',
                       default=[512, 256, 128],
                       help='MLP hidden dimensions')
    parser.add_argument('--mlp_dropout', type=float, default=0.3,
                       help='MLP dropout rate')

    # Graph
    parser.add_argument('--jaccard_threshold', type=float, default=0.2,
                       help='Jaccard similarity threshold for graph edges')
    parser.add_argument('--use_category_init', action='store_true',
                       help='Initialize embeddings with category features')

    # Loss
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')

    # Training
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--max_epochs', type=int, default=20,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')

    # Hardware
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--precision', type=str, default='16-mixed',
                       choices=['32', '16-mixed', 'bf16-mixed'],
                       help='Training precision')

    # Logging
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory')
    parser.add_argument('--exp_name', type=str, default='srca_no_llm',
                       help='Experiment name')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Checkpoint save directory')

    return parser.parse_args()


def create_category_init_features(
    mashup_categories: list,
    api_categories: list,
    embedding_dim: int,
    all_categories: list
) -> tuple:
    """
    Create initial category-based features for embeddings.

    Uses multi-hot encoding of categories projected to embedding_dim.
    """
    import torch
    import torch.nn.functional as F

    num_categories = len(all_categories)
    category_to_idx = {cat: i for i, cat in enumerate(all_categories)}

    # Create multi-hot vectors
    mashup_multihot = torch.zeros(len(mashup_categories), num_categories)
    for i, cats in enumerate(mashup_categories):
        for cat in cats:
            if cat in category_to_idx:
                mashup_multihot[i, category_to_idx[cat]] = 1.0

    api_multihot = torch.zeros(len(api_categories), num_categories)
    for i, cats in enumerate(api_categories):
        for cat in cats:
            if cat in category_to_idx:
                api_multihot[i, category_to_idx[cat]] = 1.0

    # Project to embedding_dim using random projection
    projection = torch.randn(num_categories, embedding_dim) / (num_categories ** 0.5)

    mashup_init = torch.matmul(mashup_multihot, projection)
    api_init = torch.matmul(api_multihot, projection)

    # Normalize
    mashup_init = F.normalize(mashup_init, dim=1)
    api_init = F.normalize(api_init, dim=1)

    return mashup_init, api_init


def main():
    args = parse_args()

    logger.info("="*80)
    logger.info("SRCA Model Training WITHOUT LLM Semantic Features")
    logger.info("="*80)

    # Set random seeds
    pl.seed_everything(42)

    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data module
    logger.info("\n" + "="*80)
    logger.info("Loading Data...")
    logger.info("="*80)

    datamodule = SRCADataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    datamodule.setup()

    num_mashups = datamodule.num_mashups
    num_apis = datamodule.num_apis

    logger.info(f"Dataset statistics:")
    logger.info(f"  Mashups: {num_mashups}")
    logger.info(f"  APIs: {num_apis}")
    logger.info(f"  Train samples: {len(datamodule.train_dataset)}")
    logger.info(f"  Val samples: {len(datamodule.val_dataset)}")
    logger.info(f"  Test samples: {len(datamodule.test_dataset)}")

    # Build category co-occurrence graph
    logger.info("\n" + "="*80)
    logger.info("Building Category Co-occurrence Graph...")
    logger.info("="*80)

    graph_path = os.path.join(args.data_dir, 'category_graph.pt')

    if os.path.exists(graph_path):
        logger.info(f"Loading graph from {graph_path}")
        graph_builder = MashupAPIGraph(threshold=args.jaccard_threshold)
        graph_builder.load(graph_path)
        edge_index = graph_builder.edge_index
        edge_weight = graph_builder.edge_weight
    else:
        logger.info(f"Building new graph with threshold={args.jaccard_threshold}")
        graph_builder = MashupAPIGraph(threshold=args.jaccard_threshold)
        edge_index, edge_weight = graph_builder.build_from_data(
            datamodule.mashup_categories,
            datamodule.api_categories
        )
        graph_builder.save(graph_path)

    logger.info(f"Graph: {edge_index.size(1)} edges")

    # Create category-based initialization if requested
    mashup_category_init = None
    api_category_init = None

    if args.use_category_init:
        logger.info("\n" + "="*80)
        logger.info("Creating category-based embedding initialization...")
        logger.info("="*80)

        all_categories = list(set(
            cat for cats in datamodule.mashup_categories for cat in cats
        ) | set(
            cat for cats in datamodule.api_categories for cat in cats
        ))

        logger.info(f"Total unique categories: {len(all_categories)}")

        mashup_category_init, api_category_init = create_category_init_features(
            datamodule.mashup_categories,
            datamodule.api_categories,
            args.embedding_dim,
            all_categories
        )

        logger.info(f"Mashup init features: {mashup_category_init.shape}")
        logger.info(f"API init features: {api_category_init.shape}")

    # Initialize model
    logger.info("\n" + "="*80)
    logger.info("Initializing Model...")
    logger.info("="*80)

    model = SRCANoLLM(
        num_apis=num_apis,
        num_mashups=num_mashups,
        embedding_dim=args.embedding_dim,
        gnn_num_layers=args.gnn_num_layers,
        mlp_hidden_dims=args.mlp_hidden_dims,
        mlp_dropout=args.mlp_dropout,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mashup_api_graph_edge_index=edge_index,
        mashup_api_graph_edge_weight=edge_weight,
        mashup_category_init=mashup_category_init,
        api_category_init=api_category_init
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")

    # Setup callbacks
    logger.info("\n" + "="*80)
    logger.info("Setting up Training...")
    logger.info("="*80)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=f'{args.exp_name}-{{epoch:02d}}-{{val/P@5:.4f}}',
        monitor='val/P@5',
        mode='max',
        save_top_k=3,
        save_last=True
    )

    early_stopping = EarlyStopping(
        monitor='val/P@5',
        patience=args.patience,
        mode='max',
        verbose=True
    )

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp_name
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        precision=args.precision,
        callbacks=[checkpoint_callback, early_stopping],
        logger=tb_logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=True
    )

    # Training
    logger.info("\n" + "="*80)
    logger.info("Starting Training...")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Max epochs: {args.max_epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Precision: {args.precision}")
    logger.info(f"  GPUs: {args.gpus}")

    trainer.fit(model, datamodule=datamodule)

    # Testing
    logger.info("\n" + "="*80)
    logger.info("Testing Best Model...")
    logger.info("="*80)

    trainer.test(model, datamodule=datamodule, ckpt_path='best')

    logger.info("\n" + "="*80)
    logger.info("Training Completed!")
    logger.info("="*80)
    logger.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
    logger.info(f"TensorBoard logs: {tb_logger.log_dir}")


if __name__ == "__main__":
    main()
