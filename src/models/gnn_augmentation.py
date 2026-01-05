"""GNN-based Feature Augmentation for SRCA.

Uses Graph Neural Network on mashup-API co-occurrence graph to augment
semantic features from LLM with category relationship information.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MashupAPIGNNLayer(MessagePassing):
    """
    Graph Neural Network layer for mashup-API feature augmentation.

    Implements Equation 9-12 from the SRCA paper:
    - Neighbor feature aggregation with normalized edge weights
    - Residual integration with layer normalization
    - ELU activation
    """

    def __init__(self, hidden_dim: int):
        """
        Initialize GNN layer.

        Args:
            hidden_dim: Hidden feature dimension
        """
        super().__init__(aggr='add', flow='source_to_target')

        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ELU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass implementing Equations 9-12.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connections [2, num_edges]
            edge_weight: Edge weights [num_edges]

        Returns:
            Augmented node features [num_nodes, hidden_dim]
        """
        # Store original features for residual connection (Equation 10)
        x_residual = x

        # Compute node degrees for normalization
        row, col = edge_index
        deg_row = degree(row, x.size(0), dtype=x.dtype)
        deg_col = degree(col, x.size(0), dtype=x.dtype)

        # Normalize edge weights: w_ij / sqrt(|N_i| * |N_j|)
        deg_row_inv_sqrt = deg_row.pow(-0.5)
        deg_col_inv_sqrt = deg_col.pow(-0.5)
        deg_row_inv_sqrt[deg_row_inv_sqrt == float('inf')] = 0
        deg_col_inv_sqrt[deg_col_inv_sqrt == float('inf')] = 0

        norm = deg_row_inv_sqrt[row] * deg_col_inv_sqrt[col]

        if edge_weight is not None:
            norm = norm * edge_weight

        # Equation 9: Neighbor feature aggregation
        x_aggregated = self.propagate(edge_index, x=x, norm=norm)

        # Equation 10: Residual feature integration with LayerNorm
        x_combined = self.layer_norm(x_residual + x_aggregated)

        # Equation 12: Non-linear transformation with ELU
        x_output = self.activation(x_combined)

        return x_output

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """
        Construct messages from neighbors.

        Args:
            x_j: Neighbor node features [num_edges, hidden_dim]
            norm: Normalization coefficients [num_edges]

        Returns:
            Normalized messages [num_edges, hidden_dim]
        """
        return norm.view(-1, 1) * x_j


class FeatureAugmentation(nn.Module):
    """
    Complete feature augmentation module using mashup-API category co-occurrence graph.

    Augments LLM semantic features through multi-layer GNN propagation,
    capturing both direct and high-order service relationships.
    """

    def __init__(
        self,
        semantic_dim: int,
        gnn_num_layers: int = 2
    ):
        """
        Initialize Feature Augmentation module.

        Args:
            semantic_dim: Dimension of LLM semantic features
            gnn_num_layers: Number of GNN layers
        """
        super().__init__()

        self.semantic_dim = semantic_dim
        self.gnn_num_layers = gnn_num_layers

        # GNN layers for feature augmentation
        self.gnn_layers = nn.ModuleList([
            MashupAPIGNNLayer(semantic_dim)
            for _ in range(gnn_num_layers)
        ])

        logger.info(f"Initialized FeatureAugmentation with {gnn_num_layers} GNN layers")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Augment semantic features through GNN propagation.

        Args:
            x: Service semantic features [num_services, semantic_dim]
               (mashups + APIs concatenated)
            edge_index: Graph structure [2, num_edges]
            edge_weight: Edge weights [num_edges]

        Returns:
            Augmented features [num_services, semantic_dim]
        """
        # Multi-layer GNN propagation (no dropout - relies on LayerNorm for regularization)
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_weight)

        return x


def test_feature_augmentation():
    """Test feature augmentation module."""
    print("Testing Feature Augmentation...")

    num_mashups = 5
    num_apis = 3
    num_services = num_mashups + num_apis
    semantic_dim = 768

    # Create random service features
    service_features = torch.randn(num_services, semantic_dim)

    # Create a simple bipartite graph (mashups connect to APIs)
    edges = []
    weights = []

    # Connect each mashup to some APIs
    for m in range(num_mashups):
        for a in range(num_apis):
            if (m + a) % 2 == 0:  # Simple pattern
                a_idx = num_mashups + a
                # Mashup -> API
                edges.append([m, a_idx])
                weights.append(0.5)
                # API -> Mashup
                edges.append([a_idx, m])
                weights.append(0.5)

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_weight = torch.tensor(weights, dtype=torch.float)

    # Initialize module
    augmentation = FeatureAugmentation(
        semantic_dim=semantic_dim,
        gnn_num_layers=2
    )

    # Forward pass
    print("\nRunning forward pass...")
    augmented_features = augmentation(
        x=service_features,
        edge_index=edge_index,
        edge_weight=edge_weight
    )

    print(f"✓ Input features: {service_features.shape}")
    print(f"✓ Graph: {num_mashups} mashups + {num_apis} APIs")
    print(f"✓ Edges: {edge_index.size(1)}")
    print(f"✓ Output augmented features: {augmented_features.shape}")
    print(f"✓ Feature range: [{augmented_features.min():.4f}, {augmented_features.max():.4f}]")

    assert augmented_features.shape == service_features.shape, "Output shape mismatch!"

    print("\n✓ Feature Augmentation test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_feature_augmentation()
