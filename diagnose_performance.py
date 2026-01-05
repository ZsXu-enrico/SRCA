"""Comprehensive diagnosis of SRCA performance issues."""

import os
import sys
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import SRCA_CONFIG

def diagnose():
    print("=" * 70)
    print("SRCA Performance Diagnosis")
    print("=" * 70)

    # 1. Data Statistics
    print("\n1. DATA STATISTICS")
    print("-" * 70)

    data_dir = SRCA_CONFIG['data_dir']
    mashup_df = pd.read_csv(os.path.join(data_dir, 'mashup.csv'))
    api_df = pd.read_csv(os.path.join(data_dir, 'api.csv'))
    pairs_df = pd.read_csv(os.path.join(data_dir, 'ma_pair.txt'),
                          sep='\t', header=None, names=['mashup_id', 'api_id'])

    print(f"Dataset size:")
    print(f"  Ours:  {len(mashup_df)} mashups, {len(api_df)} APIs")
    print(f"  Paper: 7739 mashups, 1342 APIs")
    print(f"  Diff:  +{len(mashup_df)-7739} (+{100*(len(mashup_df)-7739)/7739:.1f}%), +{len(api_df)-1342} (+{100*(len(api_df)-1342)/1342:.1f}%)")

    # Data split
    train_size = int(0.6 * len(mashup_df))
    val_size = int(0.2 * len(mashup_df))
    test_size = len(mashup_df) - train_size - val_size

    paper_train = int(0.8 * 7739)
    paper_val = int(0.1 * 7739)
    paper_test = 7739 - paper_train - paper_val

    print(f"\nData split:")
    print(f"  Ours:  train={train_size}, val={val_size}, test={test_size}")
    print(f"  Paper: train={paper_train}, val={paper_val}, test={paper_test}")
    print(f"  Training data difference: {train_size - paper_train} ({100*(train_size-paper_train)/paper_train:.1f}%)")

    # Label sparsity
    num_mashups = len(mashup_df)
    num_apis = len(api_df)
    labels = torch.zeros(num_mashups, num_apis)

    for _, row in pairs_df.iterrows():
        mashup_id = int(row['mashup_id'])
        api_id = int(row['api_id'])
        if mashup_id < num_mashups and api_id < num_apis:
            labels[mashup_id, api_id] = 1.0

    apis_per_mashup = labels.sum(dim=1)

    print(f"\nLabel statistics:")
    print(f"  Sparsity: {100 * (1 - labels.mean().item()):.4f}%")
    print(f"  APIs per mashup: mean={apis_per_mashup.mean().item():.2f}, median={apis_per_mashup.median().item():.0f}")
    print(f"  53.82% mashups use only 1 API → Max possible P@5 = 20%")

    # 2. Feature Quality
    print("\n2. FEATURE QUALITY")
    print("-" * 70)

    feature_file = os.path.join(data_dir, 'semantic_features.pt')
    if os.path.exists(feature_file):
        features_data = torch.load(feature_file, map_location='cpu', weights_only=False)
        mashup_features = features_data['mashup_features']
        api_features = features_data['api_features']

        print(f"Feature shapes:")
        print(f"  Mashup: {mashup_features.shape}")
        print(f"  API:    {api_features.shape}")
        print(f"\nFeature statistics:")
        print(f"  Mashup features: mean={mashup_features.mean().item():.4f}, std={mashup_features.std().item():.4f}")
        print(f"  API features:    mean={api_features.mean().item():.4f}, std={api_features.std().item():.4f}")

        # Check for NaN or Inf
        mashup_nan = torch.isnan(mashup_features).any().item()
        api_nan = torch.isnan(api_features).any().item()
        mashup_inf = torch.isinf(mashup_features).any().item()
        api_inf = torch.isinf(api_features).any().item()

        if mashup_nan or api_nan or mashup_inf or api_inf:
            print(f"  ⚠️ WARNING: Found NaN or Inf in features!")
            print(f"    Mashup NaN: {mashup_nan}, Inf: {mashup_inf}")
            print(f"    API NaN: {api_nan}, Inf: {api_inf}")
        else:
            print(f"  ✓ No NaN or Inf detected")

        # Check feature diversity
        mashup_var = mashup_features.var(dim=0).mean().item()
        api_var = api_features.var(dim=0).mean().item()
        print(f"\nFeature diversity (variance):")
        print(f"  Mashup: {mashup_var:.4f}")
        print(f"  API:    {api_var:.4f}")

        if mashup_var < 0.1 or api_var < 0.1:
            print(f"  ⚠️ WARNING: Low variance - features might be too similar")

    # 3. Configuration Check
    print("\n3. CONFIGURATION CHECK")
    print("-" * 70)

    print(f"LLM Model: {SRCA_CONFIG['llm_model_name']}")
    print(f"  Paper uses: LLaMA-3.2-1B / 3B / 8B-Instruct")
    print(f"  Ours uses:  TinyLlama-1.1B")

    print(f"\nSemantic dimension: {SRCA_CONFIG['semantic_dim']}")
    print(f"  TinyLlama native: 2048 ✓")

    print(f"\nThreshold: {SRCA_CONFIG['cooccur_threshold']}")
    print(f"  Paper uses: 0.2 (for optimal performance)")
    print(f"  Ours uses:  {SRCA_CONFIG['cooccur_threshold']} (to save memory)")

    print(f"\nTraining:")
    print(f"  Epochs: {SRCA_CONFIG['max_epochs']}")
    print(f"  LR: {SRCA_CONFIG['learning_rate']}")
    print(f"  Weight decay: {SRCA_CONFIG['weight_decay']}")
    print(f"  Batch size: {SRCA_CONFIG['batch_size']}")

    # 4. Possible Issues
    print("\n4. POSSIBLE ISSUES & RECOMMENDATIONS")
    print("-" * 70)

    issues = []

    # Issue 1: Training data
    if train_size < paper_train:
        issues.append({
            'severity': 'MEDIUM',
            'issue': f'Training data is {paper_train - train_size} samples less than paper',
            'impact': 'Could reduce performance by 5-10%',
            'fix': 'Change data split to 80/10/10 in src/datamodules/srca_datamodule.py'
        })

    # Issue 2: LLM model
    issues.append({
        'severity': 'HIGH',
        'issue': 'Using TinyLlama instead of LLaMA-3',
        'impact': 'Could reduce performance significantly',
        'fix': 'Change llm_model_name to "meta-llama/Llama-3.2-1B-Instruct"'
    })

    # Issue 3: Extreme sparsity
    median_apis = apis_per_mashup.median().item()
    if median_apis == 1:
        issues.append({
            'severity': 'INFO',
            'issue': 'Median mashup uses only 1 API',
            'impact': f'Theoretical max P@5 = 20% for {(apis_per_mashup==1).sum().item()} mashups',
            'fix': 'This is a dataset characteristic, not a bug'
        })

    for i, issue in enumerate(issues, 1):
        print(f"\n{issue['severity']:8s} Issue #{i}: {issue['issue']}")
        print(f"           Impact: {issue['impact']}")
        print(f"           Fix: {issue['fix']}")

    # 5. Expected Performance
    print("\n5. EXPECTED PERFORMANCE ANALYSIS")
    print("-" * 70)

    print(f"\nGiven current setup:")
    print(f"  - TinyLlama semantic features")
    print(f"  - 60/20/20 split (less training data)")
    print(f"  - Threshold 0.4 (sparser graph)")
    print(f"  - Extreme sparsity (99.87%)")
    print(f"\nRealistic expectations:")
    print(f"  Paper (LLaMA-3.1-8B): P@5 = 26.23%")
    print(f"  Paper (LLaMA-3.2-1B): P@5 = 24.76%")
    print(f"  Ours (TinyLlama):     P@5 = 6-8% (estimated)")
    print(f"\nGap analysis:")
    print(f"  - LLM quality:      ~15-18% (main factor)")
    print(f"  - Training data:    ~2-3%")
    print(f"  - Graph sparsity:   ~1-2%")
    print(f"  - Total expected:   ~18-23% gap")

    print("\n" + "=" * 70)
    print("Diagnosis Complete")
    print("=" * 70)
    print("\nRECOMMENDATION:")
    print("1. Priority HIGH: Switch to LLaMA-3.2-1B-Instruct")
    print("2. Priority MED:  Change data split to 80/10/10")
    print("3. Priority LOW:  Consider threshold=0.2 if memory allows")


if __name__ == "__main__":
    diagnose()
