"""Simplified SRCA Model without LLM Semantic Features.

This is an ablation study model that removes LLM semantic encoding.
Instead, it uses:
- Learnable embeddings for mashups and APIs
- Category co-occurrence graph (same as original SRCA)
- GNN-based feature augmentation (same as original SRCA)
- Recommendation MLP with Focal Loss (same as original SRCA)

This helps understand the contribution of LLM semantic features.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Optional
import logging
import numpy as np

from .gnn_augmentation import FeatureAugmentation
from .recommendation_mlp import RecommendationMLP
from .focal_loss import FocalLoss

logger = logging.getLogger(__name__)


class SRCANoLLM(pl.LightningModule):
    """
    Simplified SRCA model without LLM semantic features.

    Uses learnable embeddings + GNN + MLP instead of LLM + GNN + MLP.
    """

    def __init__(
        self,
        # Model structure
        num_apis: int,
        num_mashups: int,
        embedding_dim: int = 768,
        gnn_num_layers: int = 2,
        mlp_hidden_dims: List[int] = [512, 256, 128],
        mlp_dropout: float = 0.3,
        # Loss parameters
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        # Graph
        mashup_api_graph_edge_index: Optional[torch.Tensor] = None,
        mashup_api_graph_edge_weight: Optional[torch.Tensor] = None,
        # Category features (optional, for initialization)
        mashup_category_init: Optional[torch.Tensor] = None,
        api_category_init: Optional[torch.Tensor] = None,
        # Evaluation
        eval_k_values: List[int] = [1, 3, 5, 10]
    ):
        """
        Initialize SRCA model without LLM.

        Args:
            num_apis: Number of APIs
            num_mashups: Number of mashups
            embedding_dim: Embedding dimension
            gnn_num_layers: Number of GNN layers
            mlp_hidden_dims: MLP hidden layer dimensions
            mlp_dropout: Dropout rate
            focal_alpha: Focal loss alpha
            focal_gamma: Focal loss gamma
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            mashup_api_graph_edge_index: Graph structure
            mashup_api_graph_edge_weight: Graph edge weights
            mashup_category_init: Initial category features for mashups (optional)
            api_category_init: Initial category features for APIs (optional)
            eval_k_values: K values for evaluation
        """
        super().__init__()
        self.save_hyperparameters(ignore=[
            'mashup_api_graph_edge_index', 'mashup_api_graph_edge_weight',
            'mashup_category_init', 'api_category_init'
        ])

        self.num_apis = num_apis
        self.num_mashups = num_mashups
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eval_k_values = eval_k_values

        logger.info(f"Initializing SRCA-NoLLM model...")
        logger.info(f"  Mashups: {num_mashups}, APIs: {num_apis}")
        logger.info(f"  Embedding dim: {embedding_dim}")

        # Learnable embeddings (replacing LLM semantic features)
        self.mashup_embedding = nn.Embedding(num_mashups, embedding_dim)
        self.api_embedding = nn.Embedding(num_apis, embedding_dim)

        # Initialize with category features if provided
        if mashup_category_init is not None:
            logger.info("Initializing mashup embeddings with category features")
            with torch.no_grad():
                self.mashup_embedding.weight.copy_(mashup_category_init)
        else:
            # Xavier uniform initialization
            nn.init.xavier_uniform_(self.mashup_embedding.weight)

        if api_category_init is not None:
            logger.info("Initializing API embeddings with category features")
            with torch.no_grad():
                self.api_embedding.weight.copy_(api_category_init)
        else:
            nn.init.xavier_uniform_(self.api_embedding.weight)

        # Feature Augmentation (GNN)
        logger.info("Initializing Feature Augmentation (GNN)...")
        self.feature_augmentation = FeatureAugmentation(
            semantic_dim=embedding_dim,
            gnn_num_layers=gnn_num_layers,
            dropout=mlp_dropout
        )

        # Recommendation MLP
        logger.info("Initializing Recommendation MLP...")
        self.recommendation_mlp = RecommendationMLP(
            input_dim=embedding_dim,
            hidden_dims=mlp_hidden_dims,
            num_apis=num_apis,
            dropout=mlp_dropout
        )

        # Loss function
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # Register graph as buffers
        if mashup_api_graph_edge_index is not None:
            self.register_buffer('mashup_api_graph_edge_index', mashup_api_graph_edge_index)
            self.register_buffer('mashup_api_graph_edge_weight', mashup_api_graph_edge_weight)
            logger.info(f"  Graph edges: {mashup_api_graph_edge_index.size(1)}")
        else:
            self.mashup_api_graph_edge_index = None
            self.mashup_api_graph_edge_weight = None
            logger.warning("  No graph provided, using embeddings only!")

        logger.info("SRCA-NoLLM Model initialized successfully")

    def forward(self, mashup_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            mashup_indices: Batch of mashup indices [batch_size]

        Returns:
            API recommendation logits [batch_size, num_apis]
        """
        # Get all service embeddings
        all_mashup_features = self.mashup_embedding.weight  # [num_mashups, embedding_dim]
        all_api_features = self.api_embedding.weight        # [num_apis, embedding_dim]

        # Apply GNN augmentation if graph is available
        if self.mashup_api_graph_edge_index is not None:
            # Concatenate all service features
            all_service_features = torch.cat([
                all_mashup_features,
                all_api_features
            ], dim=0)  # [num_mashups + num_apis, embedding_dim]

            # Apply GNN (computes fresh gradients each time)
            augmented_features = self.feature_augmentation(
                x=all_service_features,
                edge_index=self.mashup_api_graph_edge_index,
                edge_weight=self.mashup_api_graph_edge_weight
            )

            # Extract mashup features for this batch
            batch_mashup_features = augmented_features[mashup_indices]
        else:
            # No GNN, use embeddings directly
            batch_mashup_features = all_mashup_features[mashup_indices]

        # Predict API recommendations
        logits = self.recommendation_mlp(batch_mashup_features)

        return logits

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        mashup_indices = batch['mashup_indices']
        labels = batch['labels']

        # Forward pass
        logits = self(mashup_indices)

        # Compute loss
        loss = self.criterion(logits, labels)

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        """Validation step."""
        mashup_indices = batch['mashup_indices']
        labels = batch['labels']

        # Forward pass
        logits = self(mashup_indices)

        # Compute loss
        loss = self.criterion(logits, labels)
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)

        # Compute metrics
        metrics = self.compute_metrics(logits, labels)
        for k, v in metrics.items():
            self.log(f'val/{k}', v, on_epoch=True)

    def test_step(self, batch: Dict, batch_idx: int):
        """Test step."""
        mashup_indices = batch['mashup_indices']
        labels = batch['labels']

        # Forward pass
        logits = self(mashup_indices)

        # Compute loss
        loss = self.criterion(logits, labels)
        self.log('test/loss', loss, on_epoch=True)

        # Compute metrics
        metrics = self.compute_metrics(logits, labels)
        for k, v in metrics.items():
            self.log(f'test/{k}', v, on_epoch=True)

    def compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        probs = torch.sigmoid(logits)
        metrics = {}

        for k in self.eval_k_values:
            # Get top-K predictions
            _, top_k_indices = torch.topk(probs, k, dim=1)

            # Compute Precision@K
            precision = self.precision_at_k(top_k_indices, labels, k)
            metrics[f'P@{k}'] = precision

            # Compute Recall@K
            recall = self.recall_at_k(top_k_indices, labels, k)
            metrics[f'R@{k}'] = recall

            # Compute NDCG@K
            ndcg = self.ndcg_at_k(probs, labels, k)
            metrics[f'NDCG@{k}'] = ndcg

            # Compute MAP@K
            map_score = self.map_at_k(probs, labels, k)
            metrics[f'MAP@{k}'] = map_score

        return metrics

    @staticmethod
    def precision_at_k(predictions: torch.Tensor, labels: torch.Tensor, k: int) -> float:
        """Compute Precision@K."""
        batch_size = predictions.size(0)
        hits = 0

        for i in range(batch_size):
            pred = predictions[i]
            label = labels[i]
            hits += (label[pred] == 1).sum().item()

        return hits / (batch_size * k)

    @staticmethod
    def recall_at_k(predictions: torch.Tensor, labels: torch.Tensor, k: int) -> float:
        """Compute Recall@K."""
        batch_size = predictions.size(0)
        total_recall = 0

        for i in range(batch_size):
            pred = predictions[i]
            label = labels[i]
            num_relevant = label.sum().item()

            if num_relevant > 0:
                hits = (label[pred] == 1).sum().item()
                total_recall += hits / num_relevant

        return total_recall / batch_size

    @staticmethod
    def ndcg_at_k(probs: torch.Tensor, labels: torch.Tensor, k: int) -> float:
        """Compute NDCG@K."""
        batch_size = probs.size(0)
        total_ndcg = 0

        for i in range(batch_size):
            top_k_probs, top_k_indices = torch.topk(probs[i], k)
            relevance = labels[i][top_k_indices].cpu().numpy()

            # DCG
            dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))

            # IDCG
            ideal_relevance = np.sort(labels[i].cpu().numpy())[::-1][:k]
            idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))

            if idcg > 0:
                total_ndcg += dcg / idcg

        return total_ndcg / batch_size

    @staticmethod
    def map_at_k(probs: torch.Tensor, labels: torch.Tensor, k: int) -> float:
        """Compute MAP@K."""
        batch_size = probs.size(0)
        total_ap = 0

        for i in range(batch_size):
            _, top_k_indices = torch.topk(probs[i], k)
            relevance = labels[i][top_k_indices].cpu().numpy()

            num_relevant = 0
            sum_precision = 0

            for j, rel in enumerate(relevance):
                if rel == 1:
                    num_relevant += 1
                    sum_precision += num_relevant / (j + 1)

            if num_relevant > 0:
                total_ap += sum_precision / min(num_relevant, k)

        return total_ap / batch_size

    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/P@5',
                'interval': 'epoch',
                'frequency': 1
            }
        }


def test_srca_no_llm():
    """Test SRCA-NoLLM model."""
    print("Testing SRCA-NoLLM model...")

    num_mashups = 100
    num_apis = 50
    embedding_dim = 256

    # Create dummy graph
    edges = []
    weights = []
    for m in range(num_mashups):
        for a in range(num_apis):
            if (m + a) % 3 == 0:
                # Mashup -> API
                edges.append([m, num_mashups + a])
                weights.append(0.5)
                # API -> Mashup
                edges.append([num_mashups + a, m])
                weights.append(0.5)

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_weight = torch.tensor(weights, dtype=torch.float)

    # Initialize model
    model = SRCANoLLM(
        num_apis=num_apis,
        num_mashups=num_mashups,
        embedding_dim=embedding_dim,
        gnn_num_layers=2,
        mlp_hidden_dims=[512, 256, 128],
        mashup_api_graph_edge_index=edge_index,
        mashup_api_graph_edge_weight=edge_weight
    )

    print(f"\n✓ Model initialized")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 16
    mashup_indices = torch.randint(0, num_mashups, (batch_size,))
    labels = torch.zeros(batch_size, num_apis)
    for i in range(batch_size):
        num_pos = np.random.randint(1, 5)
        pos_apis = np.random.choice(num_apis, num_pos, replace=False)
        labels[i, pos_apis] = 1.0

    batch = {
        'mashup_indices': mashup_indices,
        'labels': labels
    }

    print(f"\n✓ Testing forward pass...")
    logits = model(mashup_indices)
    print(f"  Logits shape: {logits.shape}")

    print(f"\n✓ Testing training step...")
    loss = model.training_step(batch, 0)
    print(f"  Loss: {loss.item():.4f}")

    print(f"\n✓ Testing validation step...")
    model.validation_step(batch, 0)

    print("\n✓ SRCA-NoLLM test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_srca_no_llm()
