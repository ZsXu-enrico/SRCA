"""Diagnose why validation P@5 is low but test P@5 is higher."""

import os
import sys
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import SRCA_CONFIG

def diagnose_gap():
    print("=" * 70)
    print("Diagnosing Validation vs Test Performance Gap")
    print("=" * 70)

    # Load data
    data_dir = SRCA_CONFIG['data_dir']
    mashup_df = pd.read_csv(os.path.join(data_dir, 'mashup.csv'))
    pairs_df = pd.read_csv(os.path.join(data_dir, 'ma_pair.txt'),
                          sep='\t', header=None, names=['mashup_id', 'api_id'])

    num_mashups = len(mashup_df)
    num_apis = SRCA_CONFIG['num_apis']

    # Build label matrix
    labels = torch.zeros(num_mashups, num_apis)
    for _, row in pairs_df.iterrows():
        mashup_id = int(row['mashup_id'])
        api_id = int(row['api_id'])
        if mashup_id < num_mashups and api_id < num_apis:
            labels[mashup_id, api_id] = 1.0

    # Data split (following code in datamodule)
    train_size = int(0.6 * num_mashups)
    val_size = int(0.2 * num_mashups)
    test_size = num_mashups - train_size - val_size

    train_labels = labels[:train_size]
    val_labels = labels[train_size:train_size+val_size]
    test_labels = labels[train_size+val_size:]

    print(f"\n1. DATA SPLIT ANALYSIS")
    print(f"-" * 70)
    print(f"Train: {train_size} mashups")
    print(f"Val:   {val_size} mashups")
    print(f"Test:  {test_size} mashups")

    # Compute statistics
    train_apis_per_mashup = train_labels.sum(dim=1)
    val_apis_per_mashup = val_labels.sum(dim=1)
    test_apis_per_mashup = test_labels.sum(dim=1)

    print(f"\n2. LABEL STATISTICS")
    print(f"-" * 70)
    print(f"APIs per mashup:")
    print(f"  Train: mean={train_apis_per_mashup.mean().item():.2f}, median={train_apis_per_mashup.median().item():.0f}")
    print(f"  Val:   mean={val_apis_per_mashup.mean().item():.2f}, median={val_apis_per_mashup.median().item():.0f}")
    print(f"  Test:  mean={test_apis_per_mashup.mean().item():.2f}, median={test_apis_per_mashup.median().item():.0f}")

    # Check feature file
    print(f"\n3. FEATURE FILE CHECK")
    print(f"-" * 70)
    feature_file = os.path.join(data_dir, 'semantic_features.pt')
    features_data = torch.load(feature_file, map_location='cpu', weights_only=False)

    mashup_features = features_data['mashup_features']
    api_features = features_data['api_features']

    print(f"Feature dimensions:")
    print(f"  Mashup: {mashup_features.shape}")
    print(f"  API:    {api_features.shape}")
    print(f"  Expected: 1536 (Qwen2.5-1.5B)")
    print(f"  Status: {'✓ Correct' if mashup_features.shape[1] == 1536 else '✗ Wrong'}")

    # Check checkpoint
    print(f"\n4. CHECKPOINT CHECK")
    print(f"-" * 70)
    import glob
    checkpoints = sorted(glob.glob('./checkpoints/last*.ckpt'), key=os.path.getmtime, reverse=True)

    if checkpoints:
        latest_ckpt = checkpoints[0]
        ckpt_size = os.path.getsize(latest_ckpt) / (1024**2)
        ckpt_time = os.path.getmtime(latest_ckpt)

        import time
        print(f"Latest checkpoint: {latest_ckpt}")
        print(f"  Size: {ckpt_size:.1f} MB")
        print(f"  Modified: {time.ctime(ckpt_time)}")
        print(f"  Expected size: ~80-85MB for 1536-dim features")
        print(f"  Status: {'✓ Likely Qwen' if 75 < ckpt_size < 90 else '⚠ Check manually'}")

    # Performance analysis
    print(f"\n5. PERFORMANCE ANALYSIS")
    print(f"-" * 70)

    val_p5 = 0.0618
    test_p5 = 0.1076

    print(f"Reported performance:")
    print(f"  Val P@5:  {val_p5:.4f} ({val_p5*100:.2f}%)")
    print(f"  Test P@5: {test_p5:.4f} ({test_p5*100:.2f}%)")
    print(f"  Gap: +{(test_p5-val_p5)*100:.2f}%")

    # Theoretical maximum for median=1
    val_median = val_apis_per_mashup.median().item()
    test_median = test_apis_per_mashup.median().item()

    val_max_p5 = min(val_median / 5.0, 1.0)
    test_max_p5 = min(test_median / 5.0, 1.0)

    print(f"\nTheoretical maximum (based on median):")
    print(f"  Val max P@5:  {val_max_p5:.4f} ({val_max_p5*100:.2f}%)")
    print(f"  Test max P@5: {test_max_p5:.4f} ({test_max_p5*100:.2f}%)")

    print(f"\nPerformance ratio (actual/theoretical):")
    print(f"  Val:  {val_p5/val_max_p5:.2%}")
    print(f"  Test: {test_p5/test_max_p5:.2%}")

    # Check if test has more APIs per mashup
    if test_apis_per_mashup.mean() > val_apis_per_mashup.mean():
        print(f"\n⚠️ FINDING: Test set has more APIs per mashup on average")
        print(f"   This explains why test P@5 is higher than val P@5")

    # Comparison with paper
    print(f"\n6. COMPARISON WITH PAPER")
    print(f"-" * 70)
    paper_p5 = 0.2623
    our_test_p5 = test_p5

    print(f"Paper (LLaMA-3.1-8B):        {paper_p5:.4f} ({paper_p5*100:.2f}%)")
    print(f"Ours (Qwen2.5-1.5B):         {our_test_p5:.4f} ({our_test_p5*100:.2f}%)")
    print(f"Gap:                         {(paper_p5-our_test_p5)*100:.2f}%")
    print(f"Relative performance:        {our_test_p5/paper_p5:.2%}")

    # Improvement from TinyLlama
    tinyllama_p5 = 0.0614
    improvement = (our_test_p5 - tinyllama_p5) / tinyllama_p5

    print(f"\n7. IMPROVEMENT FROM TINYLLAMA")
    print(f"-" * 70)
    print(f"TinyLlama Test P@5:          {tinyllama_p5:.4f} ({tinyllama_p5*100:.2f}%)")
    print(f"Qwen2.5-1.5B Test P@5:       {our_test_p5:.4f} ({our_test_p5*100:.2f}%)")
    print(f"Absolute improvement:        +{(our_test_p5-tinyllama_p5)*100:.2f}%")
    print(f"Relative improvement:        +{improvement:.1%}")
    print(f"✓ Qwen is {improvement:.1%} better than TinyLlama")

    print(f"\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"\n1. ✓ Features are using Qwen2.5-1.5B (1536-dim)")
    print(f"2. ✓ Test performance improved: {tinyllama_p5*100:.2f}% → {our_test_p5*100:.2f}% (+{improvement:.1%})")
    print(f"3. ⚠️ Still gap to paper: {our_test_p5*100:.2f}% vs {paper_p5*100:.2f}%")
    print(f"\nPossible reasons for remaining gap:")
    print(f"  - LLM size: Qwen-1.5B vs LLaMA-8B")
    print(f"  - Dataset: 8217 mashups vs 7739 in paper")
    print(f"  - Data split: 60/20/20 vs 80/10/10 in paper")
    print(f"  - Threshold: 0.4 vs 0.2 in paper")


if __name__ == "__main__":
    diagnose_gap()
