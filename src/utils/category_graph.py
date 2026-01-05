"""Mashup-API Category Co-occurrence Graph Construction for SRCA.

Builds a bipartite graph where:
- Nodes: Mashup services + API services
- Edges: Mashup-API connections based on category Jaccard similarity
- Edge weights: Jaccard coefficient of category sets
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class MashupAPIGraph:
    """
    Constructs and manages mashup-API category co-occurrence graph.

    The graph captures relationships between mashups and APIs based on
    their category overlap using Jaccard similarity.
    """

    def __init__(self, threshold: float = 0.2):
        """
        Initialize mashup-API graph builder.

        Args:
            threshold: Minimum Jaccard similarity to create an edge
        """
        self.threshold = threshold
        self.edge_index = None
        self.edge_weight = None
        self.num_mashups = 0
        self.num_apis = 0

    def build_from_data(
        self,
        mashup_categories: List[List[str]],
        api_categories: List[List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build mashup-API bipartite graph from category data.

        Args:
            mashup_categories: List of category lists for each mashup
                              e.g., [["Social", "Tools"], ["Weather"], ...]
            api_categories: List of category lists for each API
                           e.g., [["Mapping"], ["Social", "Media"], ...]

        Returns:
            edge_index: [2, num_edges] tensor of edge connections
            edge_weight: [num_edges] tensor of edge weights (Jaccard similarities)
        """
        logger.info("Building Mashup-API category co-occurrence graph...")

        self.num_mashups = len(mashup_categories)
        self.num_apis = len(api_categories)
        total_nodes = self.num_mashups + self.num_apis

        logger.info(f"Graph structure: {self.num_mashups} mashups + {self.num_apis} APIs = {total_nodes} nodes")

        # Build edges based on category Jaccard similarity
        edges = []
        weights = []

        for m_idx, m_cats in enumerate(mashup_categories):
            m_set = set(m_cats)

            for a_idx, a_cats in enumerate(api_categories):
                a_set = set(a_cats)

                # Calculate Jaccard similarity
                intersection = len(m_set & a_set)
                union = len(m_set | a_set)

                if union > 0:
                    jaccard = intersection / union

                    if jaccard >= self.threshold:
                        # API node index: num_mashups + a_idx
                        api_node_idx = self.num_mashups + a_idx

                        # Add bidirectional edges
                        # Mashup -> API
                        edges.append([m_idx, api_node_idx])
                        weights.append(jaccard)

                        # API -> Mashup
                        edges.append([api_node_idx, m_idx])
                        weights.append(jaccard)

        if len(edges) == 0:
            # No edges meet threshold, create empty graph
            logger.warning(f"No edges meet threshold {self.threshold}, creating empty graph")
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_weight = torch.zeros(0, dtype=torch.float)
        else:
            self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            self.edge_weight = torch.tensor(weights, dtype=torch.float)

        logger.info(f"Built graph with {self.edge_index.size(1)} edges")
        if self.edge_weight.numel() > 0:
            logger.info(f"Edge weight range: [{self.edge_weight.min():.4f}, {self.edge_weight.max():.4f}]")
            logger.info(f"Mean edge weight: {self.edge_weight.mean():.4f}")

        # Add self-loops with weight 1.0
        self_loop_index = torch.arange(total_nodes, dtype=torch.long).repeat(2, 1)
        self_loop_weight = torch.ones(total_nodes, dtype=torch.float)

        self.edge_index = torch.cat([self.edge_index, self_loop_index], dim=1)
        self.edge_weight = torch.cat([self.edge_weight, self_loop_weight], dim=0)

        logger.info(f"Added self-loops, total edges: {self.edge_index.size(1)}")

        return self.edge_index, self.edge_weight

    def save(self, path: str):
        """Save graph to disk."""
        torch.save({
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weight,
            'num_mashups': self.num_mashups,
            'num_apis': self.num_apis,
            'threshold': self.threshold
        }, path)
        logger.info(f"Saved mashup-API graph to {path}")

    def load(self, path: str):
        """Load graph from disk."""
        data = torch.load(path, weights_only=False)
        self.edge_index = data['edge_index']
        self.edge_weight = data['edge_weight']
        self.num_mashups = data['num_mashups']
        self.num_apis = data['num_apis']
        self.threshold = data['threshold']
        logger.info(f"Loaded mashup-API graph from {path}")
        logger.info(f"  Mashups: {self.num_mashups}, APIs: {self.num_apis}, Edges: {self.edge_index.size(1)}")


def test_mashup_api_graph():
    """Test mashup-API co-occurrence graph construction."""
    print("Testing Mashup-API Co-occurrence Graph...")

    # Sample data
    mashup_categories = [
        ["Social", "Media", "Tools"],
        ["Social", "Communication"],
        ["Weather", "Visualization"],
        ["Weather", "Data"],
        ["Social", "Media"],
    ]

    api_categories = [
        ["Social", "Media"],  # Should connect to mashups 0, 1, 4
        ["Weather"],          # Should connect to mashups 2, 3
        ["Tools"],            # Should connect to mashup 0
        ["Mapping"],          # No strong connections
    ]

    # Build graph
    graph_builder = MashupAPIGraph(threshold=0.1)
    edge_index, edge_weight = graph_builder.build_from_data(
        mashup_categories,
        api_categories
    )

    print(f"\n✓ Built graph with {graph_builder.num_mashups} mashups + {graph_builder.num_apis} APIs")
    print(f"✓ Graph has {edge_index.size(1)} edges")
    print(f"✓ Edge weights: min={edge_weight.min():.4f}, max={edge_weight.max():.4f}")

    # Analyze connectivity
    num_edges_no_selfloop = (edge_index[0] != edge_index[1]).sum().item()
    print(f"✓ Non-self-loop edges: {num_edges_no_selfloop}")

    print("\n✓ Mashup-API Graph test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mashup_api_graph()
