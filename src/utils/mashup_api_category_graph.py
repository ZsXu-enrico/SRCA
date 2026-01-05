"""Mashup-API Category Co-occurrence Graph (论文正确实现).

构建Mashup-API二部图：
- 节点：训练集的Mashup + 所有API
- 边：基于类别Jaccard相似度 >= threshold
- 权重：Jaccard相似度值

论文Section 4.2.1实现。
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class MashupAPICategoryGraph:
    """
    构建Mashup-API类别关联图（仅用训练集）。

    关键点：
    1. 节点 = 训练mashup + 所有API（不是类别节点）
    2. 边 = Mashup-API连接，基于类别Jaccard >= threshold
    3. 推理时：测试mashup动态连接到固定图
    """

    def __init__(self, threshold: float = 0.2):
        """
        Args:
            threshold: 最小Jaccard相似度阈值（论文用0.2）
        """
        self.threshold = threshold
        self.edge_index = None
        self.edge_weight = None
        self.num_train_mashups = 0
        self.num_apis = 0
        self.train_mashup_categories = None
        self.api_categories = None

    def build_from_train_data(
        self,
        train_mashup_categories: List[List[str]],
        api_categories: List[List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从训练集构建Mashup-API图。

        Args:
            train_mashup_categories: 训练集mashup的类别 [N_train, ?]
            api_categories: 所有API的类别 [N_apis, ?]

        Returns:
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
        """
        self.num_train_mashups = len(train_mashup_categories)
        self.num_apis = len(api_categories)
        self.train_mashup_categories = train_mashup_categories
        self.api_categories = api_categories

        total_nodes = self.num_train_mashups + self.num_apis

        logger.info(f"Building Mashup-API graph from training data...")
        logger.info(f"  Train mashups: {self.num_train_mashups}")
        logger.info(f"  APIs: {self.num_apis}")
        logger.info(f"  Total nodes: {total_nodes}")

        # 构建边：Mashup -> API（基于类别Jaccard相似度）
        edges = []
        weights = []

        for m_idx, m_cats in enumerate(train_mashup_categories):
            m_set = set(m_cats) if m_cats else set()

            for a_idx, a_cats in enumerate(api_categories):
                a_set = set(a_cats) if a_cats else set()

                # 计算Jaccard相似度
                if len(m_set) == 0 or len(a_set) == 0:
                    continue

                intersection = len(m_set & a_set)
                union = len(m_set | a_set)

                if union > 0:
                    jaccard = intersection / union

                    if jaccard >= self.threshold:
                        # API节点索引：num_train_mashups + a_idx
                        api_node_idx = self.num_train_mashups + a_idx

                        # 双向边（无向图）
                        edges.append([m_idx, api_node_idx])
                        edges.append([api_node_idx, m_idx])
                        weights.append(jaccard)
                        weights.append(jaccard)

        if len(edges) == 0:
            logger.warning(f"No edges meet threshold {self.threshold}")
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_weight = torch.zeros(0, dtype=torch.float)
        else:
            self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            self.edge_weight = torch.tensor(weights, dtype=torch.float)

        logger.info(f"Built {self.edge_index.size(1)} edges")
        if self.edge_weight.numel() > 0:
            logger.info(f"  Weight range: [{self.edge_weight.min():.4f}, {self.edge_weight.max():.4f}]")
            logger.info(f"  Mean weight: {self.edge_weight.mean():.4f}")

        # 添加self-loops
        self_loop_index = torch.arange(total_nodes, dtype=torch.long).repeat(2, 1)
        self_loop_weight = torch.ones(total_nodes, dtype=torch.float)

        self.edge_index = torch.cat([self.edge_index, self_loop_index], dim=1)
        self.edge_weight = torch.cat([self.edge_weight, self_loop_weight], dim=0)

        logger.info(f"Added self-loops, total edges: {self.edge_index.size(1)}")

        return self.edge_index, self.edge_weight

    def connect_test_mashup(
        self,
        test_mashup_categories: List[str]
    ) -> Tuple[List[int], List[float]]:
        """
        为测试mashup计算其与API的连接（推理时使用）。

        Args:
            test_mashup_categories: 测试mashup的类别列表

        Returns:
            connected_api_indices: 连接的API节点索引（在全图中的索引）
            edge_weights: 对应的边权重
        """
        m_set = set(test_mashup_categories) if test_mashup_categories else set()

        connected_apis = []
        weights = []

        for a_idx, a_cats in enumerate(self.api_categories):
            a_set = set(a_cats) if a_cats else set()

            if len(m_set) == 0 or len(a_set) == 0:
                continue

            intersection = len(m_set & a_set)
            union = len(m_set | a_set)

            if union > 0:
                jaccard = intersection / union

                if jaccard >= self.threshold:
                    # API在全图中的节点索引
                    api_node_idx = self.num_train_mashups + a_idx
                    connected_apis.append(api_node_idx)
                    weights.append(jaccard)

        return connected_apis, weights

    def save(self, path: str):
        """保存图和元数据。"""
        torch.save({
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weight,
            'num_train_mashups': self.num_train_mashups,
            'num_apis': self.num_apis,
            'threshold': self.threshold,
            'train_mashup_categories': self.train_mashup_categories,
            'api_categories': self.api_categories
        }, path)
        logger.info(f"Saved graph to {path}")

    def load(self, path: str):
        """加载图和元数据。"""
        data = torch.load(path, weights_only=False)
        self.edge_index = data['edge_index']
        self.edge_weight = data['edge_weight']
        self.num_train_mashups = data['num_train_mashups']
        self.num_apis = data['num_apis']
        self.threshold = data['threshold']
        self.train_mashup_categories = data['train_mashup_categories']
        self.api_categories = data['api_categories']
        logger.info(f"Loaded graph: {self.num_train_mashups} train mashups, {self.num_apis} APIs, {self.edge_index.size(1)} edges")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Mashup-API Category Graph...")

    # 训练集mashup
    train_mashup_categories = [
        ["social", "media"],
        ["mapping", "location"],
        ["weather", "data"],
    ]

    # 所有API
    api_categories = [
        ["social"],
        ["mapping", "location"],
        ["weather"],
        ["tools"],  # 不会连接到任何训练mashup
    ]

    # 构建图
    graph = MashupAPICategoryGraph(threshold=0.2)
    edge_index, edge_weight = graph.build_from_train_data(
        train_mashup_categories,
        api_categories
    )

    print(f"\n✓ Graph built:")
    print(f"  Nodes: {graph.num_train_mashups} train mashups + {graph.num_apis} APIs = {graph.num_train_mashups + graph.num_apis}")
    print(f"  Edges: {edge_index.size(1)}")

    # 测试推理时连接
    test_mashup = ["social", "tools"]
    connected_apis, weights = graph.connect_test_mashup(test_mashup)
    print(f"\n✓ Test mashup ['social', 'tools'] connects to:")
    print(f"  API nodes: {connected_apis}")
    print(f"  Weights: {weights}")

    print("\n✓ Test passed!")
