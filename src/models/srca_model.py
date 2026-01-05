"""Complete SRCA Model with PyTorch Lightning.

Integrates all SRCA components:
- LLM Semantic Encoder (TinyLlama)
- Category Co-occurrence Graph
- GNN-based Feature Augmentation
- Recommendation MLP with Focal Loss
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Optional
import logging
import numpy as np

from .llm_semantic import LLMSemanticEncoder
from .gnn_augmentation import FeatureAugmentation
from .recommendation_mlp import RecommendationMLP, SRCARecommender
from .focal_loss import FocalLoss
from ..utils.category_graph import MashupAPIGraph

logger = logging.getLogger(__name__)


class SRCALightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for SRCA model.

    Handles training, validation, and testing with proper metrics.
    """

    def __init__(
        self,
        # Model hyperparameters
        num_apis: int,
        num_mashups: int,
        semantic_dim: int = 1536,
        gnn_num_layers: int = 2,
        mlp_hidden_dims: List[int] = [512, 256, 128],
        mlp_dropout: float = 0.3,
        # Pre-extracted features
        mashup_features: Optional[torch.Tensor] = None,
        api_features: Optional[torch.Tensor] = None,
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
        Initialize SRCA Lightning Module.

        Args:
            num_apis: Number of APIs
            num_mashups: Number of mashups
            semantic_dim: Semantic feature dimension
            gnn_num_layers: Number of GNN layers
            mlp_hidden_dims: MLP hidden dimensions
            mlp_dropout: MLP dropout rate
            mashup_features: Pre-extracted mashup features [num_mashups, semantic_dim]
            api_features: Pre-extracted API features [num_apis, semantic_dim]
            focal_alpha: Focal loss alpha
            focal_gamma: Focal loss gamma
            learning_rate: Learning rate
            weight_decay: Weight decay
            mashup_api_graph_edge_index: Mashup-API graph structure
            mashup_api_graph_edge_weight: Mashup-API graph weights
            eval_k_values: K values for evaluation metrics
        """
        super().__init__()
        self.save_hyperparameters(ignore=[
            'mashup_features', 'api_features',
            'mashup_api_graph_edge_index', 'mashup_api_graph_edge_weight'
        ])

        self.num_apis = num_apis
        self.num_mashups = num_mashups
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eval_k_values = eval_k_values

        # Register pre-extracted features as buffers (won't be trained)
        if mashup_features is not None and api_features is not None:
            self.register_buffer('mashup_features', mashup_features)
            self.register_buffer('api_features', api_features)
            logger.info(f"Registered pre-extracted features:")
            logger.info(f"  - Mashup features: {mashup_features.shape}")
            logger.info(f"  - API features: {api_features.shape}")
        else:
            raise ValueError("Pre-extracted features are required!")

        # Initialize Feature Augmentation (GNN will be part of the model)
        # Paper Section 4.2.2: GNN uses LayerNorm + ELU (no dropout)
        logger.info("Initializing Feature Augmentation...")
        self.feature_augmentation = FeatureAugmentation(
            semantic_dim=semantic_dim,
            gnn_num_layers=gnn_num_layers
        )

        # Initialize Recommendation MLP
        logger.info("Initializing Recommendation MLP...")
        self.recommendation_mlp = RecommendationMLP(
            input_dim=semantic_dim,
            hidden_dims=mlp_hidden_dims,
            num_apis=num_apis,
            dropout=mlp_dropout
        )

        # Loss function
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # Register mashup-API graph as buffers
        if mashup_api_graph_edge_index is not None:
            self.register_buffer('mashup_api_graph_edge_index', mashup_api_graph_edge_index)
            self.register_buffer('mashup_api_graph_edge_weight', mashup_api_graph_edge_weight)
        else:
            self.mashup_api_graph_edge_index = None
            self.mashup_api_graph_edge_weight = None

        logger.info("SRCA Model initialized successfully")

    def compute_augmented_features(self):
        """
        Pre-compute GNN-augmented features for all services.

        This should be called once before training starts.
        Paper Section 4.2.2: Feature augmentation is applied to enhance
        semantic representations, done once to save memory.
        """
        if self.mashup_api_graph_edge_index is not None:
            logger.info("Computing GNN-augmented features for all services...")

            # Construct full graph features
            all_service_features = torch.cat([
                self.mashup_features,
                self.api_features
            ], dim=0)

            # Apply GNN augmentation once
            with torch.no_grad():  # Don't need gradients for this pre-computation
                augmented_features = self.feature_augmentation(
                    x=all_service_features,
                    edge_index=self.mashup_api_graph_edge_index,
                    edge_weight=self.mashup_api_graph_edge_weight
                )

            # Split back into mashup and API features
            num_mashups = self.mashup_features.size(0)
            augmented_mashup_features = augmented_features[:num_mashups]
            augmented_api_features = augmented_features[num_mashups:]

            # Replace original features with augmented ones
            self.register_buffer('mashup_features', augmented_mashup_features)
            self.register_buffer('api_features', augmented_api_features)

            logger.info("✓ GNN-augmented features computed and cached")
        else:
            logger.info("No graph provided, using original semantic features")

    def forward(
        self,
        mashup_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass using pre-computed augmented features.

        Args:
            mashup_indices: Batch of mashup indices [batch_size]

        Returns:
            API recommendation logits [batch_size, num_apis]
        """
        # Use pre-computed augmented features (no GNN computation here!)
        batch_mashup_features = self.mashup_features[mashup_indices]

        # Predict API recommendations
        logits = self.recommendation_mlp(batch_mashup_features)

        return logits

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step using pre-extracted features."""
        # Get mashup indices from batch
        mashup_indices = batch['mashup_indices']

        # Forward pass with indices
        logits = self(mashup_indices)

        # Compute loss
        loss = self.criterion(logits, batch['labels'])

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        """Validation step using pre-extracted features."""
        # Get mashup indices from batch
        mashup_indices = batch['mashup_indices']

        # Forward pass with indices
        logits = self(mashup_indices)

        # Compute loss
        loss = self.criterion(logits, batch['labels'])
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)

        # Compute metrics
        metrics = self.compute_metrics(logits, batch['labels'])
        for k, v in metrics.items():
            self.log(f'val/{k}', v, on_epoch=True)

    def test_step(self, batch: Dict, batch_idx: int):
        """Test step using pre-extracted features."""
        # Get mashup indices from batch
        mashup_indices = batch['mashup_indices']

        # Forward pass with indices
        logits = self(mashup_indices)

        # Compute loss
        loss = self.criterion(logits, batch['labels'])
        self.log('test/loss', loss, on_epoch=True)

        # Compute metrics
        metrics = self.compute_metrics(logits, batch['labels'])
        for k, v in metrics.items():
            self.log(f'test/{k}', v, on_epoch=True)

    def compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics: Precision, Recall, NDCG, MAP.

        Args:
            logits: Predicted logits [batch_size, num_apis]
            labels: Ground truth labels [batch_size, num_apis]

        Returns:
            Dictionary of metrics
        """
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
            # Get top-K predictions
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
            # Get top-K predictions
            _, top_k_indices = torch.topk(probs[i], k)
            relevance = labels[i][top_k_indices].cpu().numpy()

            # Average Precision
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
        """Configure optimizers (no scheduler since no validation set)."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        return optimizer
