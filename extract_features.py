"""Extract semantic features for all mashups and APIs using LLM.

This script:
1. Loads all mashup and API descriptions
2. Uses LLM with RPM/FPA prompts to extract semantic features
3. Saves features to disk for efficient training

Based on SRCA paper Section 4.1.2
"""

import os
import sys

# Set single GPU to avoid device mismatch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import pandas as pd
import pickle
from tqdm import tqdm
import logging

sys.path.insert(0, os.path.dirname(__file__))

from config import SRCA_CONFIG
from src.models.llm_semantic import LLMSemanticEncoder
from src.models.gnn_augmentation import FeatureAugmentation
from src.utils.category_graph import MashupAPIGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_all_features():
    """Extract and save semantic features for all services with GNN augmentation."""

    print("=" * 70)
    print("SRCA Feature Extraction with GNN Augmentation")
    print("=" * 70)

    # Initialize LLM encoder with LLaMA-3-8B
    print("\n[1/4] Initializing LLaMA-3-8B Encoder...")
    print(f"Model path: {SRCA_CONFIG['llm_model_path']}")

    encoder = LLMSemanticEncoder(
        model_path=SRCA_CONFIG['llm_model_path'],
        max_length=SRCA_CONFIG['llm_max_length'],
        semantic_dim=SRCA_CONFIG['semantic_dim'],
        load_in_8bit=SRCA_CONFIG['use_8bit'],
        load_in_4bit=SRCA_CONFIG['use_4bit'],
        freeze_llm=True  # Freeze LLM parameters (only extract features)
    )
    encoder.eval()

    print(f"✓ LLaMA-3-8B Encoder loaded")
    print(f"  Semantic dim: {SRCA_CONFIG['semantic_dim']}")

    # Load mashup data
    print("\n[2/5] Extracting Mashup Features...")
    mashup_file = os.path.join(SRCA_CONFIG['data_dir'], 'mashup.csv')
    mashup_df = pd.read_csv(mashup_file)

    # Parse categories
    if 'categories' in mashup_df.columns:
        mashup_df['categories'] = mashup_df['categories'].apply(
            lambda x: x.split('|') if isinstance(x, str) and x else []
        )
    else:
        mashup_df['categories'] = [[] for _ in range(len(mashup_df))]

    # Extract mashup features using RPM prompts
    mashup_features_list = []
    batch_size = 4  # Smaller batch for LLaMA-3-8B to avoid OOM

    with torch.no_grad():
        for i in tqdm(range(0, len(mashup_df), batch_size), desc="Mashups (RPM)"):
            batch_df = mashup_df.iloc[i:i+batch_size]

            names = [
                str(row['name']) if pd.notna(row['name']) else f"Mashup_{i}"
                for i, (_, row) in enumerate(batch_df.iterrows())
            ]
            descriptions = [
                str(row['description']) if pd.notna(row['description']) else ""
                for _, row in batch_df.iterrows()
            ]
            categories = [row['categories'] for _, row in batch_df.iterrows()]

            # Extract features using RPM (Requirement-focused Prompt for Mashup)
            # Paper Section 4.1.1-4.1.2: CRITICAL - Must generate unified descriptions first!
            # Paper Eq. 4: D′_m = M(concat(T_m, D_m))
            # Then extract features from unified D′_m
            features = encoder.encode_mashups(
                descriptions,
                categories,
                names=names,
                use_generation=True  # PAPER REQUIRES: Generate unified descriptions first
            )

            mashup_features_list.append(features.cpu())

            # Clear CUDA cache periodically
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

    # Concatenate all features
    mashup_features = torch.cat(mashup_features_list, dim=0)
    print(f"✓ Extracted mashup features: {mashup_features.shape}")

    # Load API data
    print("\n[3/5] Extracting API Features...")
    api_file = os.path.join(SRCA_CONFIG['data_dir'], 'api.csv')
    api_df = pd.read_csv(api_file)

    # Parse categories
    if 'categories' in api_df.columns:
        api_df['categories'] = api_df['categories'].apply(
            lambda x: x.split('|') if isinstance(x, str) and x else []
        )
    else:
        api_df['categories'] = [[] for _ in range(len(api_df))]

    # Extract API features using FPA prompts
    api_features_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(api_df), batch_size), desc="APIs (FPA)"):
            batch_df = api_df.iloc[i:i+batch_size]

            names = [
                str(row['name']) if pd.notna(row['name']) else f"API_{i}"
                for i, (_, row) in enumerate(batch_df.iterrows())
            ]
            descriptions = [
                str(row['description']) if pd.notna(row['description']) else ""
                for _, row in batch_df.iterrows()
            ]
            categories = [row['categories'] for _, row in batch_df.iterrows()]

            # Extract features using FPA (Functional-oriented Prompt for API)
            # Paper Section 4.1.1-4.1.2: CRITICAL - Must generate unified descriptions first!
            # Paper Eq. 4: D′_a = M(concat(T_a, D_a))
            # Then extract features from unified D′_a
            features = encoder.encode_apis(
                names,
                descriptions,
                categories,
                use_generation=True  # PAPER REQUIRES: Generate unified descriptions first
            )

            api_features_list.append(features.cpu())

            # Clear CUDA cache periodically
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

    # Concatenate all features
    api_features = torch.cat(api_features_list, dim=0)
    print(f"✓ Extracted API features: {api_features.shape}")

    # Save LLM semantic features (WITHOUT GNN augmentation)
    # GNN will be applied during training as part of the model
    print("\n[4/4] Saving LLM Semantic Features...")
    output_file = os.path.join(SRCA_CONFIG['data_dir'], 'semantic_features.pt')
    torch.save({
        'mashup_features': mashup_features,
        'api_features': api_features,
        'semantic_dim': SRCA_CONFIG['semantic_dim'],
        'num_mashups': len(mashup_df),
        'num_apis': len(api_df)
    }, output_file)

    print(f"✓ Saved LLM semantic features to {output_file}")
    print(f"  - Mashup features: {mashup_features.shape}")
    print(f"  - API features: {api_features.shape}")
    print(f"  - Total size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 70)
    print("LLM Semantic Feature Extraction Complete!")
    print("=" * 70)
    print("\nNote: GNN feature augmentation will be applied during training")


if __name__ == "__main__":
    extract_all_features()
