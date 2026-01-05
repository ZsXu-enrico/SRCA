"""
Preprocess ProgrammableWeb dataset for SRCA model.

This script reads the real ProgrammableWeb data and converts it into
the format required by SRCA:
- mashup.csv: mashup descriptions and categories
- api.csv: API descriptions and categories
- ma_pair.txt: mashup-API interaction pairs
"""

import os
import json
import pandas as pd
from typing import Dict, List, Tuple
import random

# Set random seed for reproducibility
random.seed(42)


def load_json(filepath: str):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def words_to_text(words: List[str]) -> str:
    """Convert list of words to text string."""
    return ' '.join(words)


def load_programmableweb_data(source_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[int, int]]]:
    """
    Load ProgrammableWeb data from JSON files.

    Returns:
        api_df: DataFrame with API data (1647 used APIs)
        mashup_df: DataFrame with Mashup data (8217 mashups)
        ma_pairs: List of (mashup_id, api_id) pairs
    """
    print("Loading ProgrammableWeb data...")

    # Load the list of actually used APIs (1647 APIs)
    used_api_names = load_json(os.path.join(source_dir, 'used_api_list.json'))
    print(f"  ✓ Loaded {len(used_api_names)} used APIs")

    # Load ALL API data
    all_api_names = load_json(os.path.join(source_dir, 'api_name.json'))
    all_api_descs = load_json(os.path.join(source_dir, 'api_description.json'))
    all_api_cats = load_json(os.path.join(source_dir, 'api_category.json'))

    # Build mapping from API name to its data
    api_name_to_data = {}
    for name, desc, cats in zip(all_api_names, all_api_descs, all_api_cats):
        api_name_str = name[0] if isinstance(name, list) else str(name)
        api_name_to_data[api_name_str] = {
            'name': api_name_str,
            'description': words_to_text(desc) if isinstance(desc, list) else str(desc),
            'categories': cats if cats else []
        }

    # Filter to only used APIs and create mapping
    api_data = []
    api_name_to_id = {}

    for i, api_name in enumerate(used_api_names):
        if api_name in api_name_to_data:
            data = api_name_to_data[api_name]
            api_data.append({
                'id': i,
                'name': data['name'],
                'description': data['description'],
                'categories': '|'.join(data['categories']) if data['categories'] else ''
            })
            api_name_to_id[api_name] = i
        else:
            # If not found, create placeholder
            api_data.append({
                'id': i,
                'name': api_name,
                'description': f'API: {api_name}',
                'categories': ''
            })
            api_name_to_id[api_name] = i

    api_df = pd.DataFrame(api_data)
    print(f"  ✓ Created {len(api_df)} API entries")

    # Load Mashup data
    mashup_names = load_json(os.path.join(source_dir, 'mashup_name.json'))
    mashup_descs = load_json(os.path.join(source_dir, 'mashup_description.json'))
    mashup_cats = load_json(os.path.join(source_dir, 'mashup_category.json'))
    mashup_used_apis = load_json(os.path.join(source_dir, 'mashup_used_api.json'))

    print(f"  ✓ Loaded {len(mashup_names)} Mashups")

    # Create Mashup DataFrame and MA pairs
    mashup_data = []
    ma_pairs = []

    for i, (name, desc, cats, used_apis) in enumerate(zip(mashup_names, mashup_descs, mashup_cats, mashup_used_apis)):
        mashup_data.append({
            'id': i,
            'name': name[0] if isinstance(name, list) else str(name),
            'description': words_to_text(desc) if isinstance(desc, list) else str(desc),
            'categories': '|'.join(cats) if cats else ''
        })

        # Add MA pairs (only for APIs in our used API list)
        if used_apis:
            for api_name in used_apis:
                if api_name in api_name_to_id:
                    ma_pairs.append((i, api_name_to_id[api_name]))

    mashup_df = pd.DataFrame(mashup_data)

    print(f"  ✓ Created {len(mashup_df)} mashups with {len(ma_pairs)} mashup-API pairs")

    return api_df, mashup_df, ma_pairs


def save_srca_format(api_df: pd.DataFrame, mashup_df: pd.DataFrame,
                     ma_pairs: List[Tuple[int, int]], output_dir: str):
    """
    Save data in SRCA format.

    Args:
        api_df: API DataFrame
        mashup_df: Mashup DataFrame
        ma_pairs: List of (mashup_id, api_id) pairs
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\nSaving SRCA format data...")

    # Save API data
    api_output = api_df[['id', 'name', 'description', 'categories']].copy()
    api_output.to_csv(os.path.join(output_dir, 'api.csv'), index=False)
    print(f"  ✓ Saved {len(api_output)} APIs to api.csv")

    # Save Mashup data
    mashup_output = mashup_df[['id', 'name', 'description', 'categories']].copy()
    mashup_output.to_csv(os.path.join(output_dir, 'mashup.csv'), index=False)
    print(f"  ✓ Saved {len(mashup_output)} Mashups to mashup.csv")

    # Save MA pairs
    with open(os.path.join(output_dir, 'ma_pair.txt'), 'w') as f:
        for mashup_id, api_id in ma_pairs:
            f.write(f"{mashup_id}\t{api_id}\n")
    print(f"  ✓ Saved {len(ma_pairs)} mashup-API pairs to ma_pair.txt")

    # Print statistics
    print("\n" + "="*70)
    print("Data Statistics:")
    print("="*70)
    print(f"Total APIs: {len(api_df)}")
    print(f"Total Mashups: {len(mashup_df)}")
    print(f"Total MA Pairs: {len(ma_pairs)}")
    print(f"Avg APIs per Mashup: {len(ma_pairs) / len(mashup_df):.2f}")

    # Count unique categories
    all_cats = set()
    for cats in api_df['categories']:
        if cats:
            all_cats.update(cats.split('|'))
    for cats in mashup_df['categories']:
        if cats:
            all_cats.update(cats.split('|'))
    print(f"Total Unique Categories: {len(all_cats)}")
    print("="*70)


def main():
    """Main preprocessing pipeline."""
    # Get paths from config
    from config import SRCA_CONFIG

    source_dir = SRCA_CONFIG['source_data_dir']
    output_dir = SRCA_CONFIG['data_dir']

    print("="*70)
    print("SRCA Data Preprocessing - ProgrammableWeb Dataset")
    print("="*70)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    api_df, mashup_df, ma_pairs = load_programmableweb_data(source_dir)

    # Save in SRCA format
    save_srca_format(api_df, mashup_df, ma_pairs, output_dir)

    print("\n✓ Preprocessing completed successfully!")
    print(f"\nNext step: Run 'python train.py' to start training")


if __name__ == "__main__":
    main()
