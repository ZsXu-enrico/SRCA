"""简单的baseline模型."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class SimpleModel(pl.LightningModule):
    """
    简单的baseline模型：
    1. Token features [50, 1536] -> Linear -> [1536]
    2. [1536] -> Linear -> [1647]
    """

    def __init__(
        self,
        num_apis: int = 1647,
        seq_len: int = 50,
        hidden_dim: int = 1536,
        learning_rate: float = 1e-4,
        # Pre-extracted token features
        train_mashup_token_features: torch.Tensor = None,
        api_token_features: torch.Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['train_mashup_token_features', 'api_token_features'])

        self.num_apis = num_apis
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # 注册token features
        if train_mashup_token_features is not None:
            self.register_buffer('train_mashup_token_features', train_mashup_token_features)
            logger.info(f"Registered train mashup tokens: {train_mashup_token_features.shape}")
        else:
            raise ValueError("train_mashup_token_features required!")

        # Token聚合层：[batch, 50, 1536] -> [batch, 1536]
        self.token_aggregation = nn.Linear(seq_len, 1)

        # 输出层：[batch, 1536] -> [batch, 1647]
        self.output_layer = nn.Linear(hidden_dim, num_apis)

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        logger.info(f"Initialized SimpleModel: [50, 1536] -> Linear(50,1) -> [1536] -> Linear(1536, {num_apis})")

    def forward(self, mashup_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            mashup_indices: [batch_size]

        Returns:
            logits: [batch_size, 1647]
        """
        # 1. 取对应的token features
        tokens = self.train_mashup_token_features[mashup_indices]  # [batch, 50, 1536]

        # 2. 聚合50个token -> 1个feature
        # [batch, 50, 1536] -> [batch, 1536, 50] -> Linear -> [batch, 1536, 1] -> squeeze -> [batch, 1536]
        tokens_transposed = tokens.transpose(1, 2)  # [batch, 1536, 50]
        aggregated = self.token_aggregation(tokens_transposed)  # [batch, 1536, 1]
        aggregated = aggregated.squeeze(-1)  # [batch, 1536]

        # 3. 输出层
        logits = self.output_layer(aggregated)  # [batch, 1647]

        return logits

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        mashup_indices = batch['mashup_indices']
        labels = batch['labels']

        logits = self(mashup_indices)
        loss = self.criterion(logits, labels)

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Dict, batch_idx: int):
        """Test step."""
        mashup_indices = batch['mashup_indices']
        labels = batch['labels']

        # 推理时：加载完整token features
        if not hasattr(self, 'all_mashup_token_features'):
            token_file = './data/ProgrammableWeb/token_features.pt'
            token_data = torch.load(token_file, map_location=self.device, weights_only=False)
            self.all_mashup_token_features = token_data['mashup_token_features']

        # 取测试mashup的tokens
        test_tokens = self.all_mashup_token_features[mashup_indices]

        # Forward (用测试tokens)
        tokens_transposed = test_tokens.transpose(1, 2)
        aggregated = self.token_aggregation(tokens_transposed).squeeze(-1)
        logits = self.output_layer(aggregated)

        loss = self.criterion(logits, labels)
        self.log('test/loss', loss, on_epoch=True)

        # 计算指标
        probs = torch.sigmoid(logits)
        batch_size = len(labels)

        for k in [1, 3, 5, 10]:
            _, top_k_indices = torch.topk(probs, k, dim=1)

            # Precision@K
            hits = 0
            for i in range(batch_size):
                hits += (labels[i, top_k_indices[i]] == 1).sum().item()
            precision = hits / (batch_size * k)
            self.log(f'test/P@{k}', precision, on_epoch=True)

            # Recall@K
            total_recall = 0
            for i in range(batch_size):
                num_relevant = labels[i].sum().item()
                if num_relevant > 0:
                    hits = (labels[i, top_k_indices[i]] == 1).sum().item()
                    total_recall += hits / num_relevant
            recall = total_recall / batch_size
            self.log(f'test/R@{k}', recall, on_epoch=True)

            # NDCG@K
            import numpy as np
            total_ndcg = 0
            for i in range(batch_size):
                top_k_probs, top_k_idx = torch.topk(probs[i], k)
                relevance = labels[i][top_k_idx].cpu().numpy()

                # DCG
                dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))

                # IDCG
                ideal_relevance = np.sort(labels[i].cpu().numpy())[::-1][:k]
                idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))

                if idcg > 0:
                    total_ndcg += dcg / idcg

            ndcg = total_ndcg / batch_size
            self.log(f'test/NDCG@{k}', ndcg, on_epoch=True)

            # MAP@K
            total_ap = 0
            for i in range(batch_size):
                _, top_k_idx = torch.topk(probs[i], k)
                relevance = labels[i][top_k_idx].cpu().numpy()

                num_relevant = 0
                sum_precision = 0

                for j, rel in enumerate(relevance):
                    if rel == 1:
                        num_relevant += 1
                        sum_precision += num_relevant / (j + 1)

                if num_relevant > 0:
                    total_ap += sum_precision / min(num_relevant, k)

            map_score = total_ap / batch_size
            self.log(f'test/MAP@{k}', map_score, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
