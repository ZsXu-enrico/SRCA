"""Debug script to check model predictions on test set."""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import SRCA_CONFIG
from src.datamodules.srca_datamodule import SRCADataModule
from src.models.srca_model import SRCALightningModule

print("Loading data...")
datamodule = SRCADataModule(
    data_dir=SRCA_CONFIG['data_dir'],
    batch_size=SRCA_CONFIG['batch_size'],
    num_workers=0  # Use 0 for debugging
)
datamodule.prepare_data()
datamodule.setup()

# Load features
feature_file = os.path.join(SRCA_CONFIG['data_dir'], 'semantic_features.pt')
features_data = torch.load(feature_file, map_location='cpu', weights_only=False)

num_mashups_used = datamodule.num_mashups
mashup_features = features_data['mashup_features'][:num_mashups_used]
api_features = features_data['api_features']

# Load graph
graph_path = os.path.join(SRCA_CONFIG['data_dir'], 'category_graph.pt')
graph_data = torch.load(graph_path, map_location='cpu', weights_only=False)
edge_index = graph_data['edge_index']
edge_weight = graph_data['edge_weight']

print(f"\nData loaded:")
print(f"  Mashup features: {mashup_features.shape}")
print(f"  API features: {api_features.shape}")
print(f"  Test samples: {len(datamodule.test_dataset)}")

# Initialize model
print("\nInitializing model...")
model = SRCALightningModule(
    num_apis=SRCA_CONFIG['num_apis'],
    num_mashups=datamodule.num_mashups,
    semantic_dim=SRCA_CONFIG['semantic_dim'],
    gnn_num_layers=SRCA_CONFIG['gnn_num_layers'],
    mlp_hidden_dims=SRCA_CONFIG['mlp_hidden_dims'],
    mlp_dropout=SRCA_CONFIG['mlp_dropout'],
    mashup_features=mashup_features,
    api_features=api_features,
    focal_alpha=SRCA_CONFIG['focal_alpha'],
    focal_gamma=SRCA_CONFIG['focal_gamma'],
    learning_rate=SRCA_CONFIG['learning_rate'],
    weight_decay=SRCA_CONFIG['weight_decay'],
    mashup_api_graph_edge_index=edge_index,
    mashup_api_graph_edge_weight=edge_weight,
    eval_k_values=[1, 3, 5, 10]
)

# Load checkpoint
checkpoint_path = './checkpoints/last-v1.ckpt'
print(f"\nLoading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Get first batch
print("\nGetting test batch...")
test_loader = datamodule.test_dataloader()
batch = next(iter(test_loader))

print(f"Batch keys: {batch.keys()}")
print(f"Mashup indices: {batch['mashup_indices'][:10]}")
print(f"Labels shape: {batch['labels'].shape}")
print(f"Labels sum per sample: {batch['labels'].sum(dim=1)[:10]}")

# Make predictions
print("\nMaking predictions...")
with torch.no_grad():
    logits = model(batch['mashup_indices'])
    probs = torch.sigmoid(logits)

print(f"Logits shape: {logits.shape}")
print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
print(f"Probs stats: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")

# Check top predictions vs ground truth for first sample
print(f"\nFirst test sample (mashup {batch['mashup_indices'][0].item()}):")
sample_probs = probs[0]
sample_labels = batch['labels'][0]

top_k = 10
top_k_probs, top_k_indices = torch.topk(sample_probs, top_k)
print(f"  Top {top_k} predictions:")
for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
    is_correct = "✓" if sample_labels[idx] == 1 else "✗"
    print(f"    {i+1}. API {idx.item()}: prob={prob.item():.4f} {is_correct}")

print(f"\n  Ground truth:")
gt_apis = sample_labels.nonzero(as_tuple=True)[0]
print(f"    Positive APIs: {gt_apis.tolist()}")
for api_idx in gt_apis:
    print(f"      API {api_idx.item()}: model prob={sample_probs[api_idx].item():.4f}")

# Compute metrics manually for first batch
print(f"\nComputing metrics for batch...")
_, top_1_indices = torch.topk(probs, 1, dim=1)
hits = 0
for i in range(len(batch['labels'])):
    pred = top_1_indices[i]
    label = batch['labels'][i]
    if label[pred] == 1:
        hits += 1

precision_at_1 = hits / len(batch['labels'])
print(f"  P@1: {precision_at_1:.4f}")
print(f"  Hits: {hits}/{len(batch['labels'])}")
