"""Extract token-level features (no pooling) for learnable aggregation.

Instead of pooling in LLM encoder, we save all token embeddings and let
the model learn how to aggregate them.
"""

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import pandas as pd
from tqdm import tqdm
import logging

sys.path.insert(0, os.path.dirname(__file__))

from config import SRCA_CONFIG
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_token_features():
    """Extract token-level features without pooling."""

    print("=" * 70)
    print("Token-Level Feature Extraction")
    print("=" * 70)

    # Settings
    max_tokens = 50  # Fixed sequence length
    model_name = SRCA_CONFIG['llm_model_name']

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Hidden dim: {SRCA_CONFIG['semantic_dim']}")

    # Initialize model
    print("\n[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map='cuda:0'
    )
    model.eval()

    hidden_size = model.config.hidden_size
    print(f"✓ Model loaded, hidden size: {hidden_size}")

    # Load mashup data
    print("\n[2/4] Extracting Mashup Token Features...")
    mashup_file = os.path.join(SRCA_CONFIG['data_dir'], 'mashup.csv')
    mashup_df = pd.read_csv(mashup_file)

    mashup_token_features = []
    batch_size = 8

    with torch.no_grad():
        for i in tqdm(range(0, len(mashup_df), batch_size), desc="Mashups"):
            batch_df = mashup_df.iloc[i:i+batch_size]

            # Get descriptions (no generation, direct encoding)
            descriptions = [
                str(row['description'])[:500] if pd.notna(row['description']) else ""
                for _, row in batch_df.iterrows()
            ]

            # Tokenize with fixed length
            inputs = tokenizer(
                descriptions,
                padding='max_length',
                truncation=True,
                max_length=max_tokens,
                return_tensors='pt'
            )
            inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

            # Get hidden states (no generation)
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

            # Save all token embeddings (no pooling)
            mashup_token_features.append(hidden_states.cpu())

    mashup_token_features = torch.cat(mashup_token_features, dim=0)
    print(f"✓ Mashup token features: {mashup_token_features.shape}")

    # Load API data
    print("\n[3/4] Extracting API Token Features...")
    api_file = os.path.join(SRCA_CONFIG['data_dir'], 'api.csv')
    api_df = pd.read_csv(api_file)

    api_token_features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(api_df), batch_size), desc="APIs"):
            batch_df = api_df.iloc[i:i+batch_size]

            # Get descriptions
            descriptions = [
                str(row['description'])[:500] if pd.notna(row['description']) else ""
                for _, row in batch_df.iterrows()
            ]

            # Tokenize
            inputs = tokenizer(
                descriptions,
                padding='max_length',
                truncation=True,
                max_length=max_tokens,
                return_tensors='pt'
            )
            inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

            # Get hidden states
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[-1]

            api_token_features.append(hidden_states.cpu())

    api_token_features = torch.cat(api_token_features, dim=0)
    print(f"✓ API token features: {api_token_features.shape}")

    # Save
    print("\n[4/4] Saving Token Features...")
    output_file = os.path.join(SRCA_CONFIG['data_dir'], 'token_features.pt')
    torch.save({
        'mashup_token_features': mashup_token_features,  # [num_mashups, max_tokens, hidden_size]
        'api_token_features': api_token_features,        # [num_apis, max_tokens, hidden_size]
        'max_tokens': max_tokens,
        'hidden_size': hidden_size,
        'num_mashups': len(mashup_df),
        'num_apis': len(api_df)
    }, output_file)

    print(f"✓ Saved to {output_file}")
    print(f"  - Mashup tokens: {mashup_token_features.shape}")
    print(f"  - API tokens: {api_token_features.shape}")
    print(f"  - File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 70)
    print("Token-Level Feature Extraction Complete!")
    print("=" * 70)


if __name__ == "__main__":
    extract_token_features()
