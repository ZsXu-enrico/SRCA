"""SRCA Model V2 with Token-Level Features and Learnable Aggregation.

Key differences from V1:
- Uses token-level features (no pre-pooling)
- Learns separate aggregation for mashup and API
- Should achieve better feature diversity
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
from .learnable_aggregation import SeparateAggregation

logger = logging.getLogger(__name__)


class SRCALightningModuleV2(pl.LightningModule):
    """
    SRCA Model V2 with learnable aggregation over token features.
    """

    def __init__(
        self,
        # Model hyperparameters
        num_apis: int,
        num_mashups: int,
        seq_len: int = 50,
        semantic_dim: int = 1536,
        gnn_num_layers: int = 2,
        mlp_hidden_dims: List[int] = [512, 512],
        mlp_dropout: float = 0.3,
        aggregation_type: str = 'attention',  # 'attention' or 'weighted'
        # Pre-extracted token features
        mashup_token_features: Optional[torch.Tensor] = None,
        api_token_features: Optional[torch.Tensor] = None,
        # Loss parameters
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        # Mashup-API graph
        mashup_api_graph_edge_index: Optional[torch.Tensor] = None,
        mashup_api_graph_edge_weight: Optional[torch.Tensor] = None,
        # Evaluation
        eval_k_values: List[int] = [1, 3, 5, 10]
    ):
        """
        Initialize SRCA V2.

        Args:
            seq_len: Sequence length (number of tokens per description)
            aggregation_type: 'attention' or 'weighted'
            mashup_token_features: [num_mashups, seq_len, semantic_dim]
            api_token_features: [num_apis, seq_len, semantic_dim]
        """
        super().__init__()
        self.save_hyperparameters(ignore=[
            'mashup_token_features', 'api_token_features',
            'mashup_api_graph_edge_index', 'mashup_api_graph_edge_weight'
        ])

        self.num_apis = num_apis
        self.num_mashups = num_mashups
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eval_k_values = eval_k_values
        self.seq_len = seq_len
        self.semantic_dim = semantic_dim

        # Register token features as buffers
        if mashup_token_features is not None and api_token_features is not None:
            self.register_buffer('mashup_token_features', mashup_token_features)
            self.register_buffer('api_token_features', api_token_features)
            logger.info(f"Registered token features:")
            logger.info(f"  - Mashup: {mashup_token_features.shape}")
            logger.info(f"  - API: {api_token_features.shape}")
        else:
            raise ValueError("Token features are required!")

        # Learnable aggregation (mashup and API have separate aggregators)
        logger.info(f"Initializing Learnable Aggregation ({aggregation_type})...")
        self.aggregation = SeparateAggregation(
            seq_len=seq_len,
            hidden_dim=semantic_dim,
            aggregation_type=aggregation_type
        )

        # Feature Augmentation (GNN)
        logger.info("Initializing Feature Augmentation...")
        self.feature_augmentation = FeatureAugmentation(
            semantic_dim=semantic_dim,
            gnn_num_layers=gnn_num_layers,
            dropout=mlp_dropout
        )

        # Recommendation MLP
        logger.info("Initializing Recommendation MLP...")
        self.recommendation_mlp = RecommendationMLP(
            input_dim=semantic_dim,
            hidden_dims=mlp_hidden_dims,
            num_apis=num_apis,
            dropout=mlp_dropout
        )

        # Loss function
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # Register graph
        if mashup_api_graph_edge_index is not None:
            self.register_buffer('mashup_api_graph_edge_index', mashup_api_graph_edge_index)
            self.register_buffer('mashup_api_graph_edge_weight', mashup_api_graph_edge_weight)
        else:
            self.mashup_api_graph_edge_index = None
            self.mashup_api_graph_edge_weight = None

        logger.info("SRCA Model V2 initialized successfully")

    def forward(self, mashup_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            mashup_indices: [batch_size]

        Returns:
            logits: [batch_size, num_apis]
        """
        # Step 1: Aggregate all token features to get fixed-size embeddings
        # This is done for ALL services (not just the batch)
        mashup_features = self.aggregation.forward_mashup(
            self.mashup_token_features
        )  # [num_mashups, semantic_dim]

        api_features = self.aggregation.forward_api(
            self.api_token_features
        )  # [num_apis, semantic_dim]

        # Step 2: Apply GNN augmentation (if graph exists)
        if self.mashup_api_graph_edge_index is not None:
            all_features = torch.cat([mashup_features, api_features], dim=0)
            augmented_features = self.feature_augmentation(
                x=all_features,
                edge_index=self.mashup_api_graph_edge_index,
                edge_weight=self.mashup_api_graph_edge_weight
            )
            batch_mashup_features = augmented_features[mashup_indices]
        else:
            batch_mashup_features = mashup_features[mashup_indices]

        # Step 3: MLP recommendation
        logits = self.recommendation_mlp(batch_mashup_features)

        return logits

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        mashup_indices = batch['mashup_indices']
        logits = self(mashup_indices)
        loss = self.criterion(logits, batch['labels'])
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        """Validation step."""
        mashup_indices = batch['mashup_indices']
        logits = self(mashup_indices)
        loss = self.criterion(logits, batch['labels'])
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)

        metrics = self.compute_metrics(logits, batch['labels'])
        for k, v in metrics.items():
            self.log(f'val/{k}', v, on_epoch=True)

    def test_step(self, batch: Dict, batch_idx: int):
        """Test step."""
        mashup_indices = batch['mashup_indices']
        logits = self(mashup_indices)
        loss = self.criterion(logits, batch['labels'])
        self.log('test/loss', loss, on_epoch=True)

        metrics = self.compute_metrics(logits, batch['labels'])
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
            _, top_k_indices = torch.topk(probs, k, dim=1)

            precision = self.precision_at_k(top_k_indices, labels, k)
            metrics[f'P@{k}'] = precision

            recall = self.recall_at_k(top_k_indices, labels, k)
            metrics[f'R@{k}'] = recall

            ndcg = self.ndcg_at_k(probs, labels, k)
            metrics[f'NDCG@{k}'] = ndcg

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

            dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))

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
        """Configure optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer
