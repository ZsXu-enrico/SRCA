"""Recommendation MLP for SRCA model.

Multi-layer perceptron that takes augmented features and predicts
API recommendations using Focal Loss for training.
"""

import torch
import torch.nn as nn
from typing import List
import logging

from .focal_loss import FocalLoss

logger = logging.getLogger(__name__)


class RecommendationMLP(nn.Module):
    """
    Multi-layer perceptron for API recommendation.

    Takes augmented mashup features and predicts probability
    distribution over APIs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_apis: int,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialize Recommendation MLP.

        Args:
            input_dim: Input feature dimension (augmented semantic features)
            hidden_dims: List of hidden layer dimensions
                        e.g., [512, 256, 128]
            num_apis: Number of APIs to recommend
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_apis = num_apis

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation (论文4.3节使用ELU)
            layers.append(nn.ELU())

            # Dropout between hidden layers (论文4.3节)
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(prev_dim, num_apis)

        # Dropout before output
        self.output_dropout = nn.Dropout(dropout)

        logger.info(f"Initialized RecommendationMLP: {input_dim} -> {hidden_dims} -> {num_apis}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Logits [batch_size, num_apis]
        """
        h = self.hidden_layers(x)
        h = self.output_dropout(h)
        logits = self.output_layer(h)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities using sigmoid.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Probabilities [batch_size, num_apis]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict_top_k(
        self,
        x: torch.Tensor,
        k: int,
        return_scores: bool = False
    ):
        """
        Predict top-K APIs.

        Args:
            x: Input features [batch_size, input_dim]
            k: Number of top APIs to return
            return_scores: Whether to return scores along with indices

        Returns:
            If return_scores=False: [batch_size, k] tensor of API indices
            If return_scores=True: (indices, scores) tuple
        """
        probs = self.predict_proba(x)
        top_k_scores, top_k_indices = torch.topk(probs, k, dim=1)

        if return_scores:
            return top_k_indices, top_k_scores
        else:
            return top_k_indices


class SRCARecommender(nn.Module):
    """
    Complete SRCA recommendation system combining:
    - LLM semantic encoder
    - Category feature augmentation
    - Recommendation MLP with Focal Loss
    """

    def __init__(
        self,
        llm_encoder,
        feature_augmentation,
        semantic_dim: int,
        hidden_dims: List[int],
        num_apis: int,
        dropout: float = 0.3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Initialize SRCA Recommender.

        Args:
            llm_encoder: LLMSemanticEncoder instance
            feature_augmentation: FeatureAugmentation instance
            semantic_dim: Semantic feature dimension
            hidden_dims: Hidden layer dimensions for MLP
            num_apis: Number of APIs
            dropout: Dropout rate
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()

        self.llm_encoder = llm_encoder
        self.feature_augmentation = feature_augmentation
        self.recommendation_mlp = RecommendationMLP(
            input_dim=semantic_dim,
            hidden_dims=hidden_dims,
            num_apis=num_apis,
            dropout=dropout
        )

        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        logger.info("Initialized SRCA Recommender")

    def forward(
        self,
        mashup_texts: List[str],
        mashup_categories: List[List[str]],
        category_features: torch.Tensor,
        category_graph_edge_index: torch.Tensor,
        category_graph_edge_weight: torch.Tensor,
        service_category_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Full forward pass through SRCA pipeline.

        Args:
            mashup_texts: List of mashup descriptions
            mashup_categories: List of category lists for each mashup
            category_features: Category embeddings
            category_graph_edge_index: Category graph structure
            category_graph_edge_weight: Category graph edge weights
            service_category_ids: Category IDs for each service

        Returns:
            API recommendation logits [batch_size, num_apis]
        """
        # 1. Extract semantic features from LLM
        semantic_features = self.llm_encoder.encode_mashups(
            mashup_texts,
            mashup_categories,
            use_generation=False  # For efficiency during training
        )

        # 2. Augment with category features
        augmented_features = self.feature_augmentation(
            semantic_features=semantic_features,
            category_features=category_features,
            category_graph_edge_index=category_graph_edge_index,
            category_graph_edge_weight=category_graph_edge_weight,
            service_category_ids=service_category_ids
        )

        # 3. Predict API recommendations
        logits = self.recommendation_mlp(augmented_features)

        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss.

        Args:
            logits: Predicted logits [batch_size, num_apis]
            targets: Ground truth binary labels [batch_size, num_apis]

        Returns:
            Loss value
        """
        return self.criterion(logits, targets)

    def recommend(
        self,
        mashup_texts: List[str],
        mashup_categories: List[List[str]],
        category_features: torch.Tensor,
        category_graph_edge_index: torch.Tensor,
        category_graph_edge_weight: torch.Tensor,
        service_category_ids: torch.Tensor,
        k: int = 5
    ):
        """
        Generate top-K API recommendations.

        Args:
            mashup_texts: Mashup descriptions
            mashup_categories: Mashup category lists
            category_features: Category embeddings
            category_graph_edge_index: Category graph
            category_graph_edge_weight: Edge weights
            service_category_ids: Category IDs
            k: Number of recommendations

        Returns:
            Top-K API indices and scores
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                mashup_texts,
                mashup_categories,
                category_features,
                category_graph_edge_index,
                category_graph_edge_weight,
                service_category_ids
            )
            indices, scores = self.recommendation_mlp.predict_top_k(
                logits,
                k,
                return_scores=True
            )
        return indices, scores


def test_recommendation_mlp():
    """Test recommendation MLP."""
    print("Testing Recommendation MLP...")

    batch_size = 16
    input_dim = 768
    num_apis = 1647
    hidden_dims = [512, 256, 128]

    # Initialize MLP
    mlp = RecommendationMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_apis=num_apis,
        dropout=0.3
    )

    # Test forward pass
    print("\n1. Testing forward pass...")
    x = torch.randn(batch_size, input_dim)
    logits = mlp(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output logits shape: {logits.shape}")

    # Test predict_proba
    print("\n2. Testing probability prediction...")
    probs = mlp.predict_proba(x)
    print(f"   Probabilities shape: {probs.shape}")
    print(f"   Prob range: [{probs.min():.4f}, {probs.max():.4f}]")

    # Test top-K prediction
    print("\n3. Testing top-K prediction...")
    k = 5
    top_k_indices, top_k_scores = mlp.predict_top_k(x, k, return_scores=True)
    print(f"   Top-{k} indices shape: {top_k_indices.shape}")
    print(f"   Top-{k} scores shape: {top_k_scores.shape}")
    print(f"   Example top-{k} APIs: {top_k_indices[0].tolist()}")
    print(f"   Example scores: {top_k_scores[0].tolist()}")

    # Test loss computation
    print("\n4. Testing loss computation with Focal Loss...")
    targets = torch.zeros(batch_size, num_apis)
    # Random positive labels (simulating sparse targets)
    for i in range(batch_size):
        num_pos = torch.randint(1, 10, (1,)).item()
        pos_indices = torch.randint(0, num_apis, (num_pos,))
        targets[i, pos_indices] = 1.0

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    loss = criterion(logits, targets)
    print(f"   Focal Loss: {loss.item():.4f}")

    # Test backward pass
    print("\n5. Testing backward pass...")
    loss.backward()
    print(f"   ✓ Gradients computed successfully")

    print("\n✓ Recommendation MLP test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_recommendation_mlp()
