"""Analyze label statistics to understand the sparsity."""

import os
import sys
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import SRCA_CONFIG

def analyze_labels():
    """Analyze label statistics."""

    print("=" * 70)
    print("Label Statistics Analysis")
    print("=" * 70)

    # Load data
    data_dir = SRCA_CONFIG['data_dir']

    print("\n[1/3] Loading data files...")
    mashup_df = pd.read_csv(os.path.join(data_dir, 'mashup.csv'))
    api_df = pd.read_csv(os.path.join(data_dir, 'api.csv'))
    pairs_df = pd.read_csv(os.path.join(data_dir, 'ma_pair.txt'), sep='\t',
                          header=None, names=['mashup_id', 'api_id'])

    num_mashups = len(mashup_df)
    num_apis = len(api_df)

    print(f"✓ {num_mashups} mashups")
    print(f"✓ {num_apis} APIs")
    print(f"✓ {len(pairs_df)} interactions")

    # Create label matrix
    print("\n[2/3] Building label matrix...")
    labels = torch.zeros(num_mashups, num_apis)

    for _, row in pairs_df.iterrows():
        mashup_id = int(row['mashup_id'])
        api_id = int(row['api_id'])
        if mashup_id < num_mashups and api_id < num_apis:
            labels[mashup_id, api_id] = 1.0

    # Overall statistics
    print("\n[3/3] Label Statistics:")
    print(f"\n  Total elements: {num_mashups * num_apis:,}")
    print(f"  Positive labels: {labels.sum().item():.0f}")
    print(f"  Sparsity: {100 * (1 - labels.mean().item()):.4f}%")
    print(f"  Positive rate: {100 * labels.mean().item():.4f}%")

    # Per-mashup statistics
    apis_per_mashup = labels.sum(dim=1)
    print(f"\n  APIs per mashup:")
    print(f"    Mean: {apis_per_mashup.mean().item():.2f}")
    print(f"    Median: {apis_per_mashup.median().item():.0f}")
    print(f"    Min: {apis_per_mashup.min().item():.0f}")
    print(f"    Max: {apis_per_mashup.max().item():.0f}")
    print(f"    Std: {apis_per_mashup.std().item():.2f}")

    # Distribution
    print(f"\n  Distribution of APIs per mashup:")
    bins = [0, 1, 2, 3, 5, 10, 20, 1000]
    for i in range(len(bins) - 1):
        count = ((apis_per_mashup >= bins[i]) & (apis_per_mashup < bins[i+1])).sum().item()
        pct = 100 * count / num_mashups
        print(f"    [{bins[i]:3d}, {bins[i+1]:3d}): {count:5d} mashups ({pct:5.2f}%)")

    # Per-API statistics
    mashups_per_api = labels.sum(dim=0)
    print(f"\n  Mashups per API:")
    print(f"    Mean: {mashups_per_api.mean().item():.2f}")
    print(f"    Median: {mashups_per_api.median().item():.0f}")
    print(f"    Min: {mashups_per_api.min().item():.0f}")
    print(f"    Max: {mashups_per_api.max().item():.0f}")
    print(f"    Std: {mashups_per_api.std().item():.2f}")

    # Check for mashups with no APIs
    no_api_mashups = (apis_per_mashup == 0).sum().item()
    print(f"\n  Mashups with 0 APIs: {no_api_mashups} ({100 * no_api_mashups / num_mashups:.2f}%)")

    # Check for APIs with no mashups
    no_mashup_apis = (mashups_per_api == 0).sum().item()
    print(f"  APIs with 0 mashups: {no_mashup_apis} ({100 * no_mashup_apis / num_apis:.2f}%)")

    # Split analysis
    print("\n  Split Analysis:")
    train_size = int(0.6 * num_mashups)
    val_size = int(0.2 * num_mashups)

    train_apis_per_mashup = labels[:train_size].sum(dim=1)
    val_apis_per_mashup = labels[train_size:train_size+val_size].sum(dim=1)
    test_apis_per_mashup = labels[train_size+val_size:].sum(dim=1)

    print(f"    Train: {train_apis_per_mashup.mean().item():.2f} APIs/mashup (median: {train_apis_per_mashup.median().item():.0f})")
    print(f"    Val:   {val_apis_per_mashup.mean().item():.2f} APIs/mashup (median: {val_apis_per_mashup.median().item():.0f})")
    print(f"    Test:  {test_apis_per_mashup.mean().item():.2f} APIs/mashup (median: {test_apis_per_mashup.median().item():.0f})")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    analyze_labels()
