"""Test script to verify the projection fix improves feature diversity."""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import pandas as pd
from config import SRCA_CONFIG
from src.models.llm_semantic import LLMSemanticEncoder

print("=" * 70)
print("Testing Feature Extraction Fix")
print("=" * 70)

# Load a small sample of mashup data
mashup_file = os.path.join(SRCA_CONFIG['data_dir'], 'mashup.csv')
mashup_df = pd.read_csv(mashup_file)

# Parse categories
if 'categories' in mashup_df.columns:
    mashup_df['categories'] = mashup_df['categories'].apply(
        lambda x: x.split('|') if isinstance(x, str) and x else []
    )
else:
    mashup_df['categories'] = [[] for _ in range(len(mashup_df))]

# Select 10 diverse samples for testing
test_indices = [0, 100, 500, 1000, 2000, 3000, 4000, 4500, 5000, 6000]
test_df = mashup_df.iloc[test_indices]

descriptions = [
    str(row['description']) if pd.notna(row['description']) else ""
    for _, row in test_df.iterrows()
]
categories = [row['categories'] for _, row in test_df.iterrows()]

print(f"\nTest samples: {len(test_indices)}")
print(f"Sample indices: {test_indices}")

# Initialize encoder with NEW code (Identity projection)
print("\n[1/2] Extracting features with NEW method (Identity)...")
encoder = LLMSemanticEncoder(
    model_name=SRCA_CONFIG['llm_model_name'],
    max_length=SRCA_CONFIG['llm_max_length'],
    semantic_dim=SRCA_CONFIG['semantic_dim'],
    freeze_llm=True,
    device_map='cuda:0'  # Force single GPU to avoid device mismatch
)
encoder.eval()

with torch.no_grad():
    new_features = encoder.encode_mashups(
        descriptions,
        categories,
        use_generation=True
    ).cpu()

print(f"✓ Extracted new features: {new_features.shape}")

# Load OLD features
print("\n[2/2] Loading OLD features...")
old_feature_file = './data/ProgrammableWeb/semantic_features.pt'
old_data = torch.load(old_feature_file, map_location='cpu', weights_only=False)
old_features = old_data['mashup_features'][test_indices]

print(f"✓ Loaded old features: {old_features.shape}")

# Compare diversity
print("\n" + "=" * 70)
print("Feature Diversity Comparison")
print("=" * 70)

def analyze_diversity(features, name):
    print(f"\n{name}:")

    # Check variance across samples for each dimension
    var_across_samples = features.var(dim=0).mean().item()
    print(f"  Mean variance across samples (per dim): {var_across_samples:.6f}")

    # Check mean variance within each sample
    var_within_samples = features.var(dim=1).mean().item()
    print(f"  Mean variance within samples: {var_within_samples:.4f}")

    # Check pairwise cosine similarities
    from torch.nn.functional import cosine_similarity
    similarities = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            sim = cosine_similarity(features[i].unsqueeze(0), features[j].unsqueeze(0))
            similarities.append(sim.item())

    mean_sim = sum(similarities) / len(similarities)
    min_sim = min(similarities)
    max_sim = max(similarities)

    print(f"  Pairwise cosine similarity:")
    print(f"    Mean: {mean_sim:.4f}")
    print(f"    Min:  {min_sim:.4f}")
    print(f"    Max:  {max_sim:.4f}")

    # Check feature norms
    norms = features.norm(dim=1)
    print(f"  Feature norms:")
    print(f"    Mean: {norms.mean().item():.4f}")
    print(f"    Std:  {norms.std().item():.4f}")
    print(f"    Min:  {norms.min().item():.4f}")
    print(f"    Max:  {norms.max().item():.4f}")

    return var_across_samples, mean_sim

old_var, old_sim = analyze_diversity(old_features, "OLD Features (Random Projection)")
new_var, new_sim = analyze_diversity(new_features, "NEW Features (Identity)")

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"\nVariance improvement: {new_var / old_var:.2f}x")
print(f"Similarity reduction: {old_sim:.4f} -> {new_sim:.4f} (Δ = {old_sim - new_sim:.4f})")

if new_var > old_var * 2:
    print("\n✓ SUCCESS: New method significantly improves feature diversity!")
    print("  Recommendation: Re-extract all features with the fixed method.")
else:
    print("\n⚠ WARNING: Improvement is marginal. Further investigation needed.")
