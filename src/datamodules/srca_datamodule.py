"""DataModule for SRCA model using SEHGN dataset.

Loads and prepares data for training SRCA model with:
- Mashup descriptions and categories
- API information
- API-Mashup interaction labels
"""

import os
import json
import pickle
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SRCADataset(Dataset):
    """
    Dataset for SRCA model.

    Each sample contains:
    - Mashup description
    - Mashup categories
    - Category IDs
    - Target API labels (multi-hot vector)
    """

    def __init__(
        self,
        mashup_data: pd.DataFrame,
        api_labels: torch.Tensor,
        category_to_id: Dict[str, int],
        max_categories: int = 10
    ):
        """
        Initialize SRCA dataset.

        Args:
            mashup_data: DataFrame with columns ['description', 'categories']
            api_labels: Binary label tensor [num_mashups, num_apis]
            category_to_id: Mapping from category name to ID
            max_categories: Maximum number of categories per mashup
        """
        self.mashup_data = mashup_data
        self.api_labels = api_labels
        self.category_to_id = category_to_id
        self.max_categories = max_categories

    def __len__(self) -> int:
        return len(self.mashup_data)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        row = self.mashup_data.iloc[idx]

        # Get mashup global index (original index before split)
        mashup_idx = self.mashup_data.index[idx]

        # Get description
        description = str(row['description']) if pd.notna(row['description']) else ""

        # Get categories
        categories = row['categories'] if isinstance(row['categories'], list) else []

        # Convert categories to IDs
        category_ids = [
            self.category_to_id[cat] for cat in categories
            if cat in self.category_to_id
        ]

        # Pad or truncate to max_categories
        if len(category_ids) < self.max_categories:
            category_ids = category_ids + [-1] * (self.max_categories - len(category_ids))
        else:
            category_ids = category_ids[:self.max_categories]

        # Get API labels
        labels = self.api_labels[idx]

        return {
            'mashup_index': mashup_idx,  # Global mashup index for feature lookup
            'description': description,
            'categories': categories,
            'category_ids': torch.tensor(category_ids, dtype=torch.long),
            'labels': labels
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for SRCA dataset.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched data dictionary
    """
    mashup_indices = torch.tensor([item['mashup_index'] for item in batch], dtype=torch.long)
    descriptions = [item['description'] for item in batch]
    categories = [item['categories'] for item in batch]
    category_ids = torch.stack([item['category_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'mashup_indices': mashup_indices,
        'descriptions': descriptions,
        'categories': categories,
        'category_ids': category_ids,
        'labels': labels
    }


class SRCADataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for SRCA.

    Loads SEHGN dataset and prepares it for training.
    """

    def __init__(
        self,
        data_dir: str = '../data/SEHGN/',
        batch_size: int = 32,
        num_workers: int = 4,
        max_categories: int = 10
    ):
        """
        Initialize SRCA DataModule.

        Args:
            data_dir: Directory containing SEHGN data
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            max_categories: Maximum categories per service
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_categories = max_categories

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.category_to_id = None

        # Store dataset statistics
        self.num_mashups = 0
        self.num_apis = 0
        self.mashup_categories = []
        self.api_categories = []

    def prepare_data(self):
        """Download or prepare data (called on single GPU)."""
        # Check if data exists
        required_files = ['ma_pair.txt', 'mashup.csv', 'api.csv']
        for file in required_files:
            filepath = os.path.join(self.data_dir, file)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Required data file not found: {filepath}")

        logger.info(f"Data directory verified: {self.data_dir}")

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets (called on every GPU).

        Args:
            stage: 'fit', 'validate', 'test', or None
        """
        logger.info("Setting up SRCA datasets...")

        # Load mashup data
        mashup_file = os.path.join(self.data_dir, 'mashup.csv')
        mashup_df = pd.read_csv(mashup_file)

        # Paper uses ALL available data (8217 mashups in our case, 7739 in paper)
        # Transductive setting: category graph includes all mashups (train+val+test)
        num_mashups = len(mashup_df)
        self.num_mashups = num_mashups
        logger.info(f"Using all {num_mashups} mashups (paper uses transductive setting)")

        # Load API data
        api_file = os.path.join(self.data_dir, 'api.csv')
        api_df = pd.read_csv(api_file)
        num_apis = len(api_df)
        self.num_apis = num_apis
        logger.info(f"Loaded {num_apis} APIs")

        # Load API-Mashup interactions
        pair_file = os.path.join(self.data_dir, 'ma_pair.txt')
        pairs = pd.read_csv(pair_file, sep='\t', header=None, names=['mashup_id', 'api_id'])
        logger.info(f"Loaded {len(pairs)} API-Mashup pairs")

        # Create binary label matrix (only for the mashups we're using)
        labels = torch.zeros(num_mashups, num_apis)

        for _, row in pairs.iterrows():
            mashup_id = int(row['mashup_id'])
            api_id = int(row['api_id'])
            if mashup_id < num_mashups and api_id < num_apis:
                labels[mashup_id, api_id] = 1.0

        logger.info(f"Created label matrix: {labels.shape}")
        logger.info(f"Positive rate: {labels.mean():.4f}")

        # Load categories from separate JSON files
        # Category files are in /home/zxu298/TRELLIS/api/data/
        # We need to go up from data_dir (./data/ProgrammableWeb/) to find them
        # Compute absolute path
        abs_data_dir = os.path.abspath(self.data_dir)
        # Go up two levels: ProgrammableWeb -> data -> SRCA, then to api/data
        srca_dir = os.path.dirname(os.path.dirname(abs_data_dir))  # /path/to/SRCA
        api_dir = os.path.dirname(srca_dir)  # /path/to/api
        source_data_dir = os.path.join(api_dir, 'data')  # /path/to/api/data

        mashup_cat_file = os.path.join(source_data_dir, 'mashup_category.json')
        api_cat_file = os.path.join(source_data_dir, 'api_category.json')
        api_name_file = os.path.join(source_data_dir, 'api_name.json')
        used_api_file = os.path.join(source_data_dir, 'used_api_list.json')

        # Load mashup categories (all mashups for transductive graph)
        if os.path.exists(mashup_cat_file):
            with open(mashup_cat_file, 'r') as f:
                all_mashup_categories = json.load(f)
            # Use all available mashup categories (up to num_mashups)
            self.mashup_categories = all_mashup_categories[:num_mashups]
            logger.info(f"Loaded {len(self.mashup_categories)} mashup categories")
        else:
            # Fallback: try to parse from mashup_df if column exists
            if 'categories' in mashup_df.columns:
                mashup_df['categories'] = mashup_df['categories'].apply(
                    lambda x: x.split('|') if isinstance(x, str) and x else []
                )
                self.mashup_categories = mashup_df['categories'].tolist()
            else:
                self.mashup_categories = [[] for _ in range(num_mashups)]
            logger.warning(f"Mashup category file not found at {mashup_cat_file}, using fallback")

        # Load API categories (need to map from full 23518 APIs to used 1647 APIs)
        if os.path.exists(api_cat_file) and os.path.exists(api_name_file) and os.path.exists(used_api_file):
            # Load all API names (23518 total)
            with open(api_name_file, 'r') as f:
                all_api_names = json.load(f)

            # Load all API categories (23518 total, same order as api_names)
            with open(api_cat_file, 'r') as f:
                all_api_categories = json.load(f)

            # Load used API names (1647 used APIs)
            with open(used_api_file, 'r') as f:
                used_api_names = json.load(f)

            # Build name-to-index mapping for all APIs
            api_name_to_idx = {name: idx for idx, name in enumerate(all_api_names)}

            # Extract categories for used APIs only
            self.api_categories = []
            for used_name in used_api_names:
                if used_name in api_name_to_idx:
                    idx = api_name_to_idx[used_name]
                    self.api_categories.append(all_api_categories[idx])
                else:
                    self.api_categories.append([])  # Empty if not found
                    logger.warning(f"API '{used_name}' not found in api_name.json")

            logger.info(f"Loaded {len(self.api_categories)} API categories (from {len(all_api_categories)} total)")
        else:
            # Fallback: try to parse from api_df if column exists
            if 'categories' in api_df.columns:
                api_df['categories'] = api_df['categories'].apply(
                    lambda x: x.split('|') if isinstance(x, str) and x else []
                )
                self.api_categories = api_df['categories'].tolist()
            else:
                self.api_categories = [[] for _ in range(num_apis)]
            logger.warning(f"API category files not found, using fallback")

        # Add categories to dataframes for dataset use
        mashup_df['categories'] = self.mashup_categories[:len(mashup_df)]

        # Build category vocabulary from both mashups and APIs
        all_categories = set()
        for cats in self.mashup_categories:
            if isinstance(cats, list):
                all_categories.update(cats)
        for cats in self.api_categories:
            if isinstance(cats, list):
                all_categories.update(cats)

        self.category_to_id = {cat: idx for idx, cat in enumerate(sorted(all_categories))}
        logger.info(f"Found {len(self.category_to_id)} unique categories")

        # Paper Section 5.1: Split into 80/10/10 for train/val/test
        train_size = int(num_mashups * 0.8)   # 80% for train = ~6574 (8217 * 0.8)
        val_size = int(num_mashups * 0.1)     # 10% for validation = ~821
        test_size = num_mashups - train_size - val_size  # 10% for test = ~822

        train_mashup_df = mashup_df.iloc[:train_size]
        train_labels = labels[:train_size]

        # Validation set (10%)
        val_mashup_df = mashup_df.iloc[train_size:train_size + val_size]
        val_labels = labels[train_size:train_size + val_size]

        # Test set (remaining ~10%)
        test_mashup_df = mashup_df.iloc[train_size + val_size:]
        test_labels = labels[train_size + val_size:]

        # Create datasets
        self.train_dataset = SRCADataset(
            train_mashup_df,
            train_labels,
            self.category_to_id,
            self.max_categories
        )

        self.val_dataset = SRCADataset(
            val_mashup_df,
            val_labels,
            self.category_to_id,
            self.max_categories
        )

        self.test_dataset = SRCADataset(
            test_mashup_df,
            test_labels,
            self.category_to_id,
            self.max_categories
        )

        logger.info(f"Dataset split:")
        logger.info(f"  Train: {len(self.train_dataset)} samples")
        logger.info(f"  Val: {len(self.val_dataset)} samples")
        logger.info(f"  Test: {len(self.test_dataset)} samples")

        # Save category mapping
        category_map_path = os.path.join(self.data_dir, 'category_to_id.pkl')
        with open(category_map_path, 'wb') as f:
            pickle.dump(self.category_to_id, f)
        logger.info(f"Saved category mapping to {category_map_path}")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )


def test_datamodule():
    """Test SRCA datamodule."""
    print("Testing SRCA DataModule...")

    # Initialize datamodule
    datamodule = SRCADataModule(
        data_dir='../data/SEHGN/',
        batch_size=4,
        num_workers=0
    )

    # Prepare and setup
    print("\nPreparing data...")
    datamodule.prepare_data()

    print("\nSetting up datasets...")
    datamodule.setup()

    # Test train dataloader
    print("\nTesting train dataloader...")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    print(f"  Batch keys: {batch.keys()}")
    print(f"  Descriptions (first): {batch['descriptions'][0][:100]}...")
    print(f"  Categories (first): {batch['categories'][0]}")
    print(f"  Category IDs shape: {batch['category_ids'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    print(f"  Positive labels: {batch['labels'].sum(dim=1).tolist()}")

    print("\n✓ SRCA DataModule test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_datamodule()
