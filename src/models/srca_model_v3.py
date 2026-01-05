"""SRCA Model V3 - 论文正确实现.

关键点：
1. Token-level features + learnable aggregation
2. Mashup-API服务图（不是类别图）
3. GNN在服务节点上传播
4. 推理时测试mashup动态连接到图
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

from .gnn_augmentation import FeatureAugmentation
from .recommendation_mlp import RecommendationMLP
from .focal_loss import FocalLoss
from .learnable_aggregation import SeparateAggregation

logger = logging.getLogger(__name__)


class SRCALightningModuleV3(pl.LightningModule):
    """
    SRCA V3 - 完整论文实现。

    架构：
    1. Token-level LLM features + learnable aggregation
    2. Mashup-API二部图（基于类别Jaccard）
    3. GNN在服务节点上传播特征
    """

    def __init__(
        self,
        # Model hyperparameters
        num_apis: int,
        num_train_mashups: int,
        seq_len: int = 50,
        semantic_dim: int = 1536,
        gnn_num_layers: int = 2,
        mlp_hidden_dims: List[int] = [512, 512],
        mlp_dropout: float = 0.3,
        aggregation_type: str = 'attention',
        # Pre-extracted token features
        train_mashup_token_features: Optional[torch.Tensor] = None,
        api_token_features: Optional[torch.Tensor] = None,
        # Loss parameters
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        # Mashup-API graph (只包含训练集)
        mashup_api_graph_edge_index: Optional[torch.Tensor] = None,
        mashup_api_graph_edge_weight: Optional[torch.Tensor] = None,
        # Evaluation
        eval_k_values: List[int] = [1, 3, 5, 10]
    ):
        """
        初始化SRCA V3。

        Args:
            num_train_mashups: 训练集mashup数量（如4000）
            train_mashup_token_features: [num_train_mashups, seq_len, semantic_dim]
            api_token_features: [num_apis, seq_len, semantic_dim]
            mashup_api_graph: 只包含训练mashup的图
        """
        super().__init__()
        self.save_hyperparameters(ignore=[
            'train_mashup_token_features', 'api_token_features',
            'mashup_api_graph_edge_index', 'mashup_api_graph_edge_weight'
        ])

        self.num_apis = num_apis
        self.num_train_mashups = num_train_mashups
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eval_k_values = eval_k_values
        self.seq_len = seq_len
        self.semantic_dim = semantic_dim

        # 注册训练集token features
        if train_mashup_token_features is not None and api_token_features is not None:
            self.register_buffer('train_mashup_token_features', train_mashup_token_features)
            self.register_buffer('api_token_features', api_token_features)
            logger.info(f"Registered token features:")
            logger.info(f"  - Train Mashup: {train_mashup_token_features.shape}")
            logger.info(f"  - API: {api_token_features.shape}")
        else:
            raise ValueError("Token features are required!")

        # Learnable aggregation
        logger.info(f"Initializing Learnable Aggregation ({aggregation_type})...")
        self.aggregation = SeparateAggregation(
            seq_len=seq_len,
            hidden_dim=semantic_dim,
            aggregation_type=aggregation_type
        )

        # GNN (在服务节点上传播)
        logger.info("Initializing GNN for service nodes...")
        self.gnn = FeatureAugmentation(
            semantic_dim=semantic_dim,
            gnn_num_layers=gnn_num_layers
        )

        # Recommendation MLP
        logger.info("Initializing Recommendation MLP...")
        self.recommendation_mlp = RecommendationMLP(
            input_dim=semantic_dim,
            hidden_dims=mlp_hidden_dims,
            num_apis=num_apis,
            dropout=mlp_dropout
        )

        # Loss
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # 注册图（只包含训练mashup）
        if mashup_api_graph_edge_index is not None:
            self.register_buffer('train_graph_edge_index', mashup_api_graph_edge_index)
            self.register_buffer('train_graph_edge_weight', mashup_api_graph_edge_weight)
        else:
            self.train_graph_edge_index = None
            self.train_graph_edge_weight = None

        logger.info("SRCA V3 initialized successfully")

    def aggregate_service_features(
        self,
        mashup_tokens: Optional[torch.Tensor] = None,
        use_train_mashups: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        聚合服务的token features。

        Args:
            mashup_tokens: 如果是测试mashup，传入其tokens；如果是训练，为None
            use_train_mashups: 是否使用训练集mashup

        Returns:
            mashup_features: [num_mashups, semantic_dim]
            api_features: [num_apis, semantic_dim]
        """
        # 聚合API features（所有API共享）
        api_features = self.aggregation.forward_api(
            self.api_token_features
        )  # [num_apis, semantic_dim]

        if use_train_mashups:
            # 训练时：聚合训练集mashup features
            mashup_features = self.aggregation.forward_mashup(
                self.train_mashup_token_features
            )  # [num_train_mashups, semantic_dim]
        else:
            # 推理时：聚合测试mashup features
            if mashup_tokens is None:
                raise ValueError("Must provide mashup_tokens for test mashups")
            mashup_features = self.aggregation.forward_mashup(
                mashup_tokens
            )  # [batch_size, semantic_dim]

        return mashup_features, api_features

    def forward(
        self,
        mashup_indices: torch.Tensor,
        is_training: bool = True,
        test_mashup_tokens: Optional[torch.Tensor] = None,
        test_mashup_categories: Optional[List[List[str]]] = None,
        graph_builder: Optional['MashupAPICategoryGraph'] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            mashup_indices: 训练集中的mashup索引 [batch_size]
            is_training: 是否训练模式
            test_mashup_tokens: 测试mashup的tokens [batch_size, seq_len, semantic_dim]
            test_mashup_categories: 测试mashup的类别列表（用于动态连接）
            graph_builder: 图构建器（用于计算测试mashup的边）

        Returns:
            logits: [batch_size, num_apis]
        """
        if is_training:
            # 训练模式：使用训练图
            # 1. 聚合所有服务的features
            train_mashup_features, api_features = self.aggregate_service_features(
                use_train_mashups=True
            )

            # 2. 拼接成完整图的节点特征
            # [num_train_mashups + num_apis, semantic_dim]
            all_service_features = torch.cat([
                train_mashup_features,
                api_features
            ], dim=0)

            # 3. GNN传播
            if self.train_graph_edge_index is not None:
                augmented_features = self.gnn(
                    x=all_service_features,
                    edge_index=self.train_graph_edge_index,
                    edge_weight=self.train_graph_edge_weight
                )
            else:
                augmented_features = all_service_features

            # 4. 取batch中mashup的特征
            batch_mashup_features = augmented_features[mashup_indices]

        else:
            # 推理模式：测试mashup动态连接到图
            if test_mashup_tokens is None or test_mashup_categories is None:
                raise ValueError("Must provide test_mashup_tokens and test_mashup_categories in test mode")

            batch_size = test_mashup_tokens.size(0)

            # 1. 聚合训练图的所有节点特征
            train_mashup_features, api_features = self.aggregate_service_features(
                use_train_mashups=True
            )
            all_train_features = torch.cat([train_mashup_features, api_features], dim=0)

            # 2. 聚合测试mashup特征
            test_mashup_features = self.aggregation.forward_mashup(
                test_mashup_tokens
            )  # [batch_size, semantic_dim]

            # 3. 对每个测试mashup，动态连接到图并GNN增强
            batch_enhanced_features = []

            for i in range(batch_size):
                # 3a. 根据类别计算连接的API
                if graph_builder is not None:
                    connected_api_indices, edge_weights = graph_builder.connect_test_mashup(
                        test_mashup_categories[i]
                    )
                else:
                    # 没有graph builder，直接用未增强特征
                    batch_enhanced_features.append(test_mashup_features[i])
                    continue

                if len(connected_api_indices) == 0:
                    # 没有连接，用未增强特征
                    batch_enhanced_features.append(test_mashup_features[i])
                    continue

                # 3b. 构建临时扩展图：[训练图节点 + 当前测试mashup]
                # 新节点索引：num_train_mashups + num_apis
                test_node_idx = self.num_train_mashups + self.num_apis

                # 扩展节点特征
                extended_features = torch.cat([
                    all_train_features,
                    test_mashup_features[i:i+1]
                ], dim=0)  # [num_train_nodes + 1, semantic_dim]

                # 构建扩展边：原图边 + 测试mashup的边
                # 测试mashup <-> connected APIs
                new_edges = []
                new_weights = []
                for api_idx, weight in zip(connected_api_indices, edge_weights):
                    # Test mashup -> API
                    new_edges.append([test_node_idx, api_idx])
                    new_weights.append(weight)
                    # API -> Test mashup
                    new_edges.append([api_idx, test_node_idx])
                    new_weights.append(weight)

                # Self-loop for test mashup
                new_edges.append([test_node_idx, test_node_idx])
                new_weights.append(1.0)

                new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
                new_edge_weight = torch.tensor(new_weights, dtype=torch.float)

                # 合并原图边和新边
                extended_edge_index = torch.cat([
                    self.train_graph_edge_index,
                    new_edge_index.to(self.train_graph_edge_index.device)
                ], dim=1)
                extended_edge_weight = torch.cat([
                    self.train_graph_edge_weight,
                    new_edge_weight.to(self.train_graph_edge_weight.device)
                ], dim=0)

                # 3c. GNN传播
                extended_augmented = self.gnn(
                    x=extended_features,
                    edge_index=extended_edge_index,
                    edge_weight=extended_edge_weight
                )

                # 3d. 取测试mashup的增强特征
                batch_enhanced_features.append(extended_augmented[test_node_idx])

            batch_mashup_features = torch.stack(batch_enhanced_features)

        # 5. MLP推荐
        logits = self.recommendation_mlp(batch_mashup_features)

        return logits

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        mashup_indices = batch['mashup_indices']
        logits = self(mashup_indices, is_training=True)
        loss = self.criterion(logits, batch['labels'])
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Dict, batch_idx: int):
        """Test step - 模拟真实inference场景."""
        # 获取测试mashup的信息
        mashup_indices = batch['mashup_indices']  # 全局索引（4000-4999）
        categories = batch['categories']  # 类别列表

        # 获取测试mashup的token features
        # 注意：mashup_indices是全局索引，需要加载完整token文件
        # 这里简化为使用预加载的features（实际部署时现场提取）

        # 从全局token features中取测试mashup的tokens
        # （需要在模型初始化时保存所有token features的引用）
        if not hasattr(self, 'all_mashup_token_features'):
            # 第一次test时加载完整token features
            token_file = './data/ProgrammableWeb/token_features.pt'
            token_data = torch.load(token_file, map_location=self.device, weights_only=False)
            self.all_mashup_token_features = token_data['mashup_token_features']

        # 取当前batch测试mashup的tokens
        test_tokens = self.all_mashup_token_features[mashup_indices]

        # 加载graph builder（用于动态连接）
        if not hasattr(self, 'graph_builder'):
            from ..utils.mashup_api_category_graph import MashupAPICategoryGraph
            graph_path = './data/ProgrammableWeb/mashup_api_graph_train_only.pt'
            self.graph_builder = MashupAPICategoryGraph()
            self.graph_builder.load(graph_path)

        # 推理模式：动态连接测试mashup到图
        logits = self(
            mashup_indices=mashup_indices,  # 不使用（inference模式）
            is_training=False,
            test_mashup_tokens=test_tokens,
            test_mashup_categories=categories,
            graph_builder=self.graph_builder
        )

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
