"""Test if features and labels are properly aligned in the dataset."""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import SRCA_CONFIG
from src.datamodules.srca_datamodule import SRCADataModule

def test_alignment():
    """Test feature-label alignment."""

    print("=" * 70)
    print("Testing Feature-Label Alignment")
    print("=" * 70)

    # Initialize datamodule
    print("\n[1/3] Loading data...")
    datamodule = SRCADataModule(
        data_dir=SRCA_CONFIG['data_dir'],
        batch_size=4,
        num_workers=0
    )

    datamodule.prepare_data()
    datamodule.setup()

    print(f"✓ Train: {len(datamodule.train_dataset)} samples")
    print(f"✓ Val: {len(datamodule.val_dataset)} samples")
    print(f"✓ Test: {len(datamodule.test_dataset)} samples")

    # Load features
    print("\n[2/3] Loading features...")
    feature_file = os.path.join(SRCA_CONFIG['data_dir'], 'semantic_features.pt')
    features_data = torch.load(feature_file, map_location='cpu', weights_only=False)
    mashup_features = features_data['mashup_features']
    print(f"✓ Loaded mashup features: {mashup_features.shape}")

    # Check alignment for validation set
    print("\n[3/3] Checking validation set alignment...")

    val_loader = datamodule.val_dataloader()
    batch = next(iter(val_loader))

    print(f"\nFirst validation batch:")
    print(f"  Mashup indices (global): {batch['mashup_indices'].tolist()}")
    print(f"  Labels shape: {batch['labels'].shape}")
    print(f"  Positive labels per sample: {batch['labels'].sum(dim=1).tolist()}")

    # Check if indices are reasonable
    train_size = int(0.6 * SRCA_CONFIG['num_mashups'])
    val_size = int(0.2 * SRCA_CONFIG['num_mashups'])

    print(f"\nExpected validation index range: [{train_size}, {train_size + val_size})")
    print(f"Actual first indices: {batch['mashup_indices'][:10].tolist()}")

    # Check a specific example
    idx_0 = batch['mashup_indices'][0].item()
    print(f"\n  Sample 0 uses feature index: {idx_0}")
    print(f"  Sample 0 has {batch['labels'][0].sum().item():.0f} positive labels")

    # Verify consistency across multiple batches
    print("\nChecking index progression across batches...")
    all_indices = []
    for i, batch in enumerate(val_loader):
        all_indices.extend(batch['mashup_indices'].tolist())
        if i >= 2:  # Check first 3 batches
            break

    print(f"  First 20 indices: {all_indices[:20]}")
    print(f"  Min index: {min(all_indices)}, Max index: {max(all_indices)}")

    # Check if indices are within expected range
    if min(all_indices) >= train_size and max(all_indices) < train_size + val_size:
        print("\n  ✓ Indices are in correct range for validation set!")
    else:
        print(f"\n  ✗ WARNING: Indices out of expected range!")
        print(f"    Expected: [{train_size}, {train_size + val_size})")
        print(f"    Actual: [{min(all_indices)}, {max(all_indices)}]")

    # Check for duplicates or gaps
    sorted_indices = sorted(set(all_indices))
    if len(sorted_indices) == len(all_indices):
        print("  ✓ No duplicate indices")
    else:
        print(f"  ✗ WARNING: {len(all_indices) - len(sorted_indices)} duplicate indices found!")

    print("\n" + "=" * 70)
    print("Alignment Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_alignment()
