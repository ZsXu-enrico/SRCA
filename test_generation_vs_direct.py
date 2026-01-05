"""Compare feature diversity: generation vs direct encoding."""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import pandas as pd
from config import SRCA_CONFIG
from src.models.llm_semantic import LLMSemanticEncoder

print("=" * 70)
print("Comparing Generation vs Direct Encoding")
print("=" * 70)

# Load diverse mashup samples
mashup_file = os.path.join(SRCA_CONFIG['data_dir'], 'mashup.csv')
mashup_df = pd.read_csv(mashup_file)

if 'categories' in mashup_df.columns:
    mashup_df['categories'] = mashup_df['categories'].apply(
        lambda x: x.split('|') if isinstance(x, str) and x else []
    )
else:
    mashup_df['categories'] = [[] for _ in range(len(mashup_df))]

# Select diverse samples
test_indices = [0, 100, 500, 1000, 2000, 3000, 4000, 4500, 5000, 6000]
test_df = mashup_df.iloc[test_indices]

descriptions = [
    str(row['description']) if pd.notna(row['description']) else ""
    for _, row in test_df.iterrows()
]
categories = [row['categories'] for _, row in test_df.iterrows()]

print(f"\nTest samples: {len(test_indices)}")
print(f"Indices: {test_indices}\n")

# Initialize encoder
print("Initializing Qwen encoder...")
encoder = LLMSemanticEncoder(
    model_name=SRCA_CONFIG['llm_model_name'],
    max_length=SRCA_CONFIG['llm_max_length'],
    semantic_dim=SRCA_CONFIG['semantic_dim'],
    freeze_llm=True,
    device_map='cuda:0'
)
encoder.eval()

print(f"✓ Model: {SRCA_CONFIG['llm_model_name']}")
print(f"✓ Using projection: {encoder.projection is not None}\n")

# Method 1: With Generation (SRCA original)
print("=" * 70)
print("[1/2] Extracting with Generation (use_generation=True)")
print("=" * 70)
with torch.no_grad():
    features_gen = encoder.encode_mashups(
        descriptions,
        categories,
        use_generation=True
    ).cpu()
print(f"✓ Extracted: {features_gen.shape}")

# Method 2: Direct Encoding (no generation)
print("\n" + "=" * 70)
print("[2/2] Extracting with Direct Encoding (use_generation=False)")
print("=" * 70)
with torch.no_grad():
    features_direct = encoder.encode_mashups(
        descriptions,
        categories,
        use_generation=False
    ).cpu()
print(f"✓ Extracted: {features_direct.shape}")

# Compare diversity
def analyze_diversity(features, name):
    print(f"\n{name}:")

    # Variance across samples
    var_across = features.var(dim=0).mean().item()
    print(f"  Variance across samples: {var_across:.6f}")

    # Variance within samples
    var_within = features.var(dim=1).mean().item()
    print(f"  Variance within samples: {var_within:.4f}")

    # Cosine similarities
    from torch.nn.functional import cosine_similarity
    similarities = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            sim = cosine_similarity(features[i].unsqueeze(0), features[j].unsqueeze(0))
            similarities.append(sim.item())

    mean_sim = sum(similarities) / len(similarities)
    min_sim = min(similarities)
    max_sim = max(similarities)

    print(f"  Cosine similarity - Mean: {mean_sim:.4f}, Min: {min_sim:.4f}, Max: {max_sim:.4f}")

    # Norms
    norms = features.norm(dim=1)
    print(f"  Norms - Mean: {norms.mean():.2f}, Std: {norms.std():.2f}")

    return var_across, mean_sim

print("\n" + "=" * 70)
print("Diversity Analysis")
print("=" * 70)

var_gen, sim_gen = analyze_diversity(features_gen, "With Generation (SRCA)")
var_dir, sim_dir = analyze_diversity(features_direct, "Direct Encoding")

# Summary
print("\n" + "=" * 70)
print("Comparison Summary")
print("=" * 70)

print(f"\nVariance across samples:")
print(f"  Generation:  {var_gen:.6f}")
print(f"  Direct:      {var_dir:.6f}")
print(f"  Improvement: {var_dir/var_gen:.2f}x" if var_dir > var_gen else f"  Worse: {var_gen/var_dir:.2f}x")

print(f"\nCosine similarity:")
print(f"  Generation:  {sim_gen:.4f}")
print(f"  Direct:      {sim_dir:.4f}")
print(f"  Change:      {sim_gen - sim_dir:+.4f} ({'better' if sim_dir < sim_gen else 'worse'})")

# Recommendation
print("\n" + "=" * 70)
print("Recommendation")
print("=" * 70)

if var_dir > var_gen * 1.5:
    print("\n✓ RECOMMENDATION: Use Direct Encoding (use_generation=False)")
    print("  Reason: Significantly better feature diversity")
elif var_dir > var_gen * 1.1:
    print("\n⚠ RECOMMENDATION: Consider Direct Encoding (use_generation=False)")
    print("  Reason: Moderately better feature diversity")
else:
    print("\n→ RECOMMENDATION: Keep Generation (use_generation=True)")
    print("  Reason: Direct encoding doesn't provide significant improvement")
    print("  Note: The high similarity may be inherent to this dataset")

print("\n" + "=" * 70)
