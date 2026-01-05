"""简单的DataModule，直接从../data加载."""

import json
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    """简单的Dataset."""

    def __init__(self, start_idx: int, end_idx: int, labels: torch.Tensor):
        """
        Args:
            start_idx: 起始mashup索引
            end_idx: 结束mashup索引（不包含）
            labels: 标签矩阵 [num_total_mashups, num_apis]
        """
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.labels = labels

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx: int) -> Dict:
        # 全局索引
        global_idx = self.start_idx + idx

        return {
            'mashup_index': global_idx,
            'labels': self.labels[global_idx]
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function."""
    mashup_indices = torch.tensor([item['mashup_index'] for item in batch], dtype=torch.long)
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'mashup_indices': mashup_indices,
        'labels': labels
    }


class SimpleDataModule(pl.LightningDataModule):
    """简单的DataModule."""

    def __init__(
        self,
        data_dir: str = '../data/',
        batch_size: int = 64,
        num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Setup datasets."""
        logger.info("Setting up simple datasets...")

        # 1. 读取mashup_used_api.json
        mashup_used_api_file = f'{self.data_dir}/mashup_used_api.json'
        with open(mashup_used_api_file, 'r') as f:
            mashup_used_api = json.load(f)  # List[List[str]]

        num_total_mashups = len(mashup_used_api)
        logger.info(f"Total mashups: {num_total_mashups}")

        # 2. 读取used_api_list.json
        used_api_file = f'{self.data_dir}/used_api_list.json'
        with open(used_api_file, 'r') as f:
            used_api_list = json.load(f)  # List[str]

        num_apis = len(used_api_list)
        logger.info(f"Total APIs: {num_apis}")

        # 3. 构建API名称到ID的映射
        api_name_to_id = {name: idx for idx, name in enumerate(used_api_list)}

        # 4. 构建标签矩阵 [num_total_mashups, num_apis]
        labels = torch.zeros(num_total_mashups, num_apis)

        for mashup_id, api_names in enumerate(mashup_used_api):
            for api_name in api_names:
                if api_name in api_name_to_id:
                    api_id = api_name_to_id[api_name]
                    labels[mashup_id, api_id] = 1.0
                else:
                    logger.warning(f"API '{api_name}' not in used_api_list")

        logger.info(f"Label matrix: {labels.shape}")
        logger.info(f"Positive rate: {labels.mean():.6f}")

        # 5. 只使用前5000个mashup（和token features对齐）
        num_mashups = 5000
        labels = labels[:num_mashups]
        logger.info(f"Using first {num_mashups} mashups")

        # 6. 4:1 划分
        train_size = int(num_mashups * 0.8)  # 4000
        test_size = num_mashups - train_size  # 1000

        logger.info(f"Split: {train_size} train, {test_size} test")

        # 7. 创建datasets
        self.train_dataset = SimpleDataset(0, train_size, labels)
        self.test_dataset = SimpleDataset(train_size, num_mashups, labels)

        # 统计
        train_labels = labels[:train_size]
        test_labels = labels[train_size:]

        logger.info(f"Train positives: {train_labels.sum().item()}")
        logger.info(f"Test positives: {test_labels.sum().item()}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
