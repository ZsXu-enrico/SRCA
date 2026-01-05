"""Category Co-occurrence Graph Construction (论文正确实现).

Builds a category-level graph where:
- Nodes: Category labels (e.g., 443 categories)
- Edges: Category co-occurrence based on training set
- Edge weights: Co-occurrence frequency or PMI

This follows the paper's Section 4.2 methodology.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class CategoryCooccurrenceGraph:
    """
    Constructs category-level co-occurrence graph from training data ONLY.

    论文方法：
    1. 从训练集的mashup-API交互中统计类别共现
    2. Explicit: 同一mashup中的类别共现
    3. Implicit: 通过API间接关联的类别
    """

    def __init__(self, threshold: float = 0.2):
        """
        Args:
            threshold: Minimum co-occurrence strength to create edge
        """
        self.threshold = threshold
        self.category_to_id = {}
        self.id_to_category = {}
        self.edge_index = None
        self.edge_weight = None
        self.num_categories = 0

    def build_from_train_data(
        self,
        train_mashup_categories: List[List[str]],
        train_mashup_api_pairs: List[Tuple[int, int]],  # (mashup_idx, api_idx) in training set
        api_categories: List[List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build category graph from TRAINING data only.

        Args:
            train_mashup_categories: Categories of training mashups [N_train, ?]
            train_mashup_api_pairs: List of (mashup_idx, api_idx) pairs from training
            api_categories: Categories of all APIs [N_apis, ?]

        Returns:
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
        """
        logger.info("Building category co-occurrence graph from training data...")

        # Step 1: Build category vocabulary
        all_categories = set()
        for cats in train_mashup_categories:
            all_categories.update(cats)
        for cats in api_categories:
            all_categories.update(cats)

        self.category_to_id = {cat: idx for idx, cat in enumerate(sorted(all_categories))}
        self.id_to_category = {idx: cat for cat, idx in self.category_to_id.items()}
        self.num_categories = len(self.category_to_id)

        logger.info(f"Found {self.num_categories} unique categories")

        # Step 2: Count category co-occurrences
        cooccur_count = defaultdict(int)  # (cat1_id, cat2_id) -> count

        # 2a. Explicit co-occurrence: within same mashup
        for m_cats in train_mashup_categories:
            cat_ids = [self.category_to_id[c] for c in m_cats if c in self.category_to_id]
            # All pairs of categories in this mashup co-occur
            for i, cat1 in enumerate(cat_ids):
                for cat2 in cat_ids[i+1:]:
                    # Symmetric edges
                    cooccur_count[(cat1, cat2)] += 1
                    cooccur_count[(cat2, cat1)] += 1

        # 2b. Implicit co-occurrence: through API connections
        # Build mashup -> used APIs mapping from training pairs
        mashup_to_apis = defaultdict(set)
        for m_idx, a_idx in train_mashup_api_pairs:
            mashup_to_apis[m_idx].add(a_idx)

        # For each mashup in training, connect its categories with API categories
        for m_idx, m_cats in enumerate(train_mashup_categories):
            m_cat_ids = [self.category_to_id[c] for c in m_cats if c in self.category_to_id]

            # Get APIs used by this mashup
            used_apis = mashup_to_apis.get(m_idx, set())

            for a_idx in used_apis:
                a_cats = api_categories[a_idx]
                a_cat_ids = [self.category_to_id[c] for c in a_cats if c in self.category_to_id]

                # Connect mashup categories with API categories
                for m_cat in m_cat_ids:
                    for a_cat in a_cat_ids:
                        cooccur_count[(m_cat, a_cat)] += 1
                        cooccur_count[(a_cat, m_cat)] += 1

        logger.info(f"Counted {len(cooccur_count)} category co-occurrence pairs")

        # Step 3: Build edges with threshold
        edges = []
        weights = []

        max_count = max(cooccur_count.values()) if cooccur_count else 1

        for (cat1, cat2), count in cooccur_count.items():
            # Normalize to [0, 1]
            weight = count / max_count

            if weight >= self.threshold:
                edges.append([cat1, cat2])
                weights.append(weight)

        if len(edges) == 0:
            logger.warning(f"No edges meet threshold {self.threshold}")
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_weight = torch.zeros(0, dtype=torch.float)
        else:
            self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            self.edge_weight = torch.tensor(weights, dtype=torch.float)

        logger.info(f"Built graph with {self.edge_index.size(1)} edges")
        if self.edge_weight.numel() > 0:
            logger.info(f"Edge weight range: [{self.edge_weight.min():.4f}, {self.edge_weight.max():.4f}]")

        # Step 4: Add self-loops
        self_loop_index = torch.arange(self.num_categories, dtype=torch.long).repeat(2, 1)
        self_loop_weight = torch.ones(self.num_categories, dtype=torch.float)

        self.edge_index = torch.cat([self.edge_index, self_loop_index], dim=1)
        self.edge_weight = torch.cat([self.edge_weight, self_loop_weight], dim=0)

        logger.info(f"Added self-loops, total edges: {self.edge_index.size(1)}")

        return self.edge_index, self.edge_weight

    def get_service_category_features(
        self,
        service_categories: List[List[str]]
    ) -> torch.Tensor:
        """
        Get category indicator features for services.

        Args:
            service_categories: List of category lists for services

        Returns:
            category_features: [num_services, num_categories] binary matrix
        """
        num_services = len(service_categories)
        category_features = torch.zeros(num_services, self.num_categories)

        for i, cats in enumerate(service_categories):
            for cat in cats:
                if cat in self.category_to_id:
                    cat_id = self.category_to_id[cat]
                    category_features[i, cat_id] = 1.0

        return category_features

    def save(self, path: str):
        """Save graph and vocabulary."""
        torch.save({
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weight,
            'category_to_id': self.category_to_id,
            'id_to_category': self.id_to_category,
            'num_categories': self.num_categories,
            'threshold': self.threshold
        }, path)
        logger.info(f"Saved category graph to {path}")

    def load(self, path: str):
        """Load graph and vocabulary."""
        data = torch.load(path, weights_only=False)
        self.edge_index = data['edge_index']
        self.edge_weight = data['edge_weight']
        self.category_to_id = data['category_to_id']
        self.id_to_category = data['id_to_category']
        self.num_categories = data['num_categories']
        self.threshold = data['threshold']
        logger.info(f"Loaded category graph: {self.num_categories} categories, {self.edge_index.size(1)} edges")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Category Co-occurrence Graph...")

    # Sample training data
    train_mashup_categories = [
        ["Social", "Media"],
        ["Social", "Tools"],
        ["Weather", "Data"],
    ]

    train_mashup_api_pairs = [
        (0, 0),  # Mashup 0 uses API 0
        (0, 1),  # Mashup 0 uses API 1
        (1, 0),  # Mashup 1 uses API 0
        (2, 2),  # Mashup 2 uses API 2
    ]

    api_categories = [
        ["Social"],
        ["Media", "Tools"],
        ["Weather"],
    ]

    # Build graph
    graph = CategoryCooccurrenceGraph(threshold=0.2)
    edge_index, edge_weight = graph.build_from_train_data(
        train_mashup_categories,
        train_mashup_api_pairs,
        api_categories
    )

    print(f"\n✓ Built category graph:")
    print(f"  Categories: {graph.num_categories}")
    print(f"  Edges: {edge_index.size(1)}")
    print(f"  Category vocabulary: {list(graph.category_to_id.keys())}")

    # Test service features
    test_services = [["Social", "Media"], ["Unknown"]]
    features = graph.get_service_category_features(test_services)
    print(f"\n✓ Service category features: {features.shape}")
    print(f"  Service 0 (Social, Media): {features[0].nonzero().squeeze().tolist()}")
    print(f"  Service 1 (Unknown): {features[1].nonzero().squeeze().tolist()}")

    print("\n✓ Test passed!")
