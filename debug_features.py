"""Debug script to check if mashup features are being accessed correctly."""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import SRCA_CONFIG
from src.datamodules.srca_datamodule import SRCADataModule

print("Loading data...")
datamodule = SRCADataModule(
    data_dir=SRCA_CONFIG['data_dir'],
    batch_size=SRCA_CONFIG['batch_size'],
    num_workers=0
)
datamodule.prepare_data()
datamodule.setup()

# Load features
feature_file = os.path.join(SRCA_CONFIG['data_dir'], 'semantic_features.pt')
features_data = torch.load(feature_file, map_location='cpu', weights_only=False)

num_mashups_used = datamodule.num_mashups
mashup_features = features_data['mashup_features'][:num_mashups_used]

print(f"\nFeature loading:")
print(f"  Total mashup features in file: {features_data['mashup_features'].shape}")
print(f"  Mashup features loaded: {mashup_features.shape}")
print(f"  DataModule num_mashups: {datamodule.num_mashups}")

# Get train and test batches
train_loader = datamodule.train_dataloader()
test_loader = datamodule.test_dataloader()

train_batch = next(iter(train_loader))
test_batch = next(iter(test_loader))

print(f"\nTrain batch:")
print(f"  Mashup indices: {train_batch['mashup_indices'][:10]}")
print(f"  Min index: {train_batch['mashup_indices'].min()}")
print(f"  Max index: {train_batch['mashup_indices'].max()}")

print(f"\nTest batch:")
print(f"  Mashup indices: {test_batch['mashup_indices'][:10]}")
print(f"  Min index: {test_batch['mashup_indices'].min()}")
print(f"  Max index: {test_batch['mashup_indices'].max()}")

# Check if we can access features for test indices
test_idx = test_batch['mashup_indices'][0].item()
print(f"\nAccessing feature for test mashup {test_idx}:")
if test_idx < len(mashup_features):
    feature = mashup_features[test_idx]
    print(f"  Feature shape: {feature.shape}")
    print(f"  Feature norm: {feature.norm().item():.4f}")
    print(f"  Feature stats: min={feature.min():.4f}, max={feature.max():.4f}, mean={feature.mean():.4f}")
else:
    print(f"  ERROR: Index {test_idx} is out of bounds for mashup_features with shape {mashup_features.shape}")

# Compare train vs test feature norms
train_indices = train_batch['mashup_indices'][:5]
test_indices = test_batch['mashup_indices'][:5]

print(f"\nFeature norms comparison:")
print(f"  Train mashups:")
for idx in train_indices:
    norm = mashup_features[idx].norm().item()
    print(f"    Mashup {idx.item()}: norm={norm:.4f}")

print(f"  Test mashups:")
for idx in test_indices:
    norm = mashup_features[idx].norm().item()
    print(f"    Mashup {idx.item()}: norm={norm:.4f}")

# Check if features are all the same (which would indicate a problem)
print(f"\nChecking feature diversity:")
feat_0 = mashup_features[0]
feat_100 = mashup_features[100]
feat_4000 = mashup_features[4000]
print(f"  Mashup 0 vs 100 cosine sim: {torch.cosine_similarity(feat_0, feat_100, dim=0).item():.4f}")
print(f"  Mashup 0 vs 4000 cosine sim: {torch.cosine_similarity(feat_0, feat_4000, dim=0).item():.4f}")
print(f"  Mashup 100 vs 4000 cosine sim: {torch.cosine_similarity(feat_100, feat_4000, dim=0).item():.4f}")
