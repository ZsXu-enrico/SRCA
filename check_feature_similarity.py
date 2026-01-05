"""Check if extracted features are too similar (low variance problem)."""

import os
import sys
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import SRCA_CONFIG

def check_feature_similarity():
    print("=" * 70)
    print("Checking Feature Similarity (Variance Analysis)")
    print("=" * 70)

    # Load features
    feature_file = os.path.join(SRCA_CONFIG['data_dir'], 'semantic_features.pt')
    features_data = torch.load(feature_file, map_location='cpu', weights_only=False)

    mashup_features = features_data['mashup_features']
    api_features = features_data['api_features']

    print(f"\nFeature shapes:")
    print(f"  Mashup: {mashup_features.shape}")
    print(f"  API:    {api_features.shape}")

    # 1. Check variance
    print("\n" + "=" * 70)
    print("1. VARIANCE ANALYSIS")
    print("=" * 70)

    mashup_var = mashup_features.var(dim=0)  # Variance per dimension
    api_var = api_features.var(dim=0)

    print(f"\nPer-dimension variance:")
    print(f"  Mashup: mean={mashup_var.mean().item():.6f}, std={mashup_var.std().item():.6f}")
    print(f"  API:    mean={api_var.mean().item():.6f}, std={api_var.std().item():.6f}")
    print(f"  Ratio (Mashup/API): {mashup_var.mean().item() / api_var.mean().item():.4f}")

    # Check low-variance dimensions
    low_var_threshold = 0.01
    mashup_low_var_dims = (mashup_var < low_var_threshold).sum().item()
    api_low_var_dims = (api_var < low_var_threshold).sum().item()

    print(f"\nDimensions with variance < {low_var_threshold}:")
    print(f"  Mashup: {mashup_low_var_dims}/{mashup_features.shape[1]} ({100*mashup_low_var_dims/mashup_features.shape[1]:.1f}%)")
    print(f"  API:    {api_low_var_dims}/{api_features.shape[1]} ({100*api_low_var_dims/api_features.shape[1]:.1f}%)")

    # 2. Check pairwise similarity
    print("\n" + "=" * 70)
    print("2. PAIRWISE SIMILARITY ANALYSIS")
    print("=" * 70)

    # Sample 1000 mashups for efficiency
    sample_size = min(1000, mashup_features.shape[0])
    sample_indices = torch.randperm(mashup_features.shape[0])[:sample_size]
    mashup_sample = mashup_features[sample_indices]

    # Compute cosine similarity matrix
    mashup_sample_norm = torch.nn.functional.normalize(mashup_sample, p=2, dim=1)
    similarity_matrix = torch.mm(mashup_sample_norm, mashup_sample_norm.t())

    # Get upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
    similarities = similarity_matrix[mask]

    print(f"\nCosine similarity statistics (sample of {sample_size} mashups):")
    print(f"  Mean: {similarities.mean().item():.4f}")
    print(f"  Median: {similarities.median().item():.4f}")
    print(f"  Std: {similarities.std().item():.4f}")
    print(f"  Min: {similarities.min().item():.4f}")
    print(f"  Max: {similarities.max().item():.4f}")

    # Check how many pairs are highly similar
    high_sim_threshold = 0.9
    high_sim_count = (similarities > high_sim_threshold).sum().item()
    total_pairs = similarities.numel()

    print(f"\nPairs with similarity > {high_sim_threshold}:")
    print(f"  {high_sim_count}/{total_pairs} ({100*high_sim_count/total_pairs:.2f}%)")

    if high_sim_count / total_pairs > 0.1:
        print(f"  ⚠️ WARNING: {100*high_sim_count/total_pairs:.1f}% of mashup pairs are highly similar!")
        print(f"     This suggests features lack diversity")

    # 3. Check feature distribution
    print("\n" + "=" * 70)
    print("3. FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    mashup_mean = mashup_features.mean(dim=0)
    mashup_std = mashup_features.std(dim=0)

    print(f"\nPer-dimension statistics:")
    print(f"  Mean of means: {mashup_mean.mean().item():.6f}")
    print(f"  Mean of stds:  {mashup_std.mean().item():.6f}")

    # Check if features are normalized
    feature_norms = torch.norm(mashup_features, p=2, dim=1)
    print(f"\nFeature L2 norms:")
    print(f"  Mean: {feature_norms.mean().item():.4f}")
    print(f"  Std:  {feature_norms.std().item():.4f}")
    print(f"  Min:  {feature_norms.min().item():.4f}")
    print(f"  Max:  {feature_norms.max().item():.4f}")

    # 4. Compare with API features
    print("\n" + "=" * 70)
    print("4. MASHUP VS API FEATURES COMPARISON")
    print("=" * 70)

    # Sample APIs
    api_sample_size = min(1000, api_features.shape[0])
    api_sample_indices = torch.randperm(api_features.shape[0])[:api_sample_size]
    api_sample = api_features[api_sample_indices]

    api_sample_norm = torch.nn.functional.normalize(api_sample, p=2, dim=1)
    api_similarity_matrix = torch.mm(api_sample_norm, api_sample_norm.t())
    api_mask = torch.triu(torch.ones_like(api_similarity_matrix), diagonal=1).bool()
    api_similarities = api_similarity_matrix[api_mask]

    print(f"\nAPI cosine similarity statistics (sample of {api_sample_size}):")
    print(f"  Mean: {api_similarities.mean().item():.4f}")
    print(f"  Std: {api_similarities.std().item():.4f}")

    print(f"\nComparison:")
    print(f"  Mashup mean similarity: {similarities.mean().item():.4f}")
    print(f"  API mean similarity:    {api_similarities.mean().item():.4f}")
    print(f"  Difference: {similarities.mean().item() - api_similarities.mean().item():.4f}")

    if similarities.mean().item() > api_similarities.mean().item() + 0.05:
        print(f"  ⚠️ WARNING: Mashup features are more similar than API features!")
        print(f"     This could explain poor recommendation performance")

    # 5. Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    issues = []

    # Check variance issue
    var_ratio = mashup_var.mean().item() / api_var.mean().item()
    if var_ratio < 0.5:
        issues.append(f"Low feature variance (ratio={var_ratio:.4f})")

    # Check similarity issue
    if high_sim_count / total_pairs > 0.05:
        issues.append(f"High feature similarity ({100*high_sim_count/total_pairs:.1f}% pairs >0.9)")

    # Check normalization
    if feature_norms.std().item() > 10:
        issues.append(f"Non-uniform feature norms (std={feature_norms.std().item():.2f})")

    if issues:
        print(f"\n⚠️ FOUND {len(issues)} ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print(f"\nPOSSIBLE CAUSES:")
        print(f"  1. TinyLlama generates template-like descriptions for mashups")
        print(f"  2. RPM prompt not diverse enough to capture mashup variety")
        print(f"  3. Mean pooling over similar generated text loses information")

        print(f"\nRECOMMENDED FIXES:")
        print(f"  1. Switch to LLaMA-3.2-1B for better generation diversity")
        print(f"  2. Check if generated descriptions are template-like")
        print(f"  3. Consider using max pooling or weighted pooling instead of mean")
    else:
        print(f"\n✓ Feature quality looks good")

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    check_feature_similarity()
