"""Check if TinyLlama generates diverse or template-like descriptions."""

import os
import sys
import torch
import pandas as pd
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, os.path.dirname(__file__))

from config import SRCA_CONFIG
from src.models.llm_semantic import LLMSemanticEncoder

def check_generation_diversity():
    print("=" * 70)
    print("Checking TinyLlama Generation Diversity")
    print("=" * 70)

    # Load encoder
    print("\n[1/3] Loading LLM...")
    encoder = LLMSemanticEncoder(
        model_name=SRCA_CONFIG['llm_model_name'],
        max_length=SRCA_CONFIG['llm_max_length'],
        semantic_dim=SRCA_CONFIG['semantic_dim'],
        freeze_llm=True
    )
    encoder.eval()
    device = next(encoder.llm.parameters()).device
    print(f"✓ Loaded on {device}")

    # Load mashup data
    print("\n[2/3] Loading sample mashups...")
    mashup_file = os.path.join(SRCA_CONFIG['data_dir'], 'mashup.csv')
    mashup_df = pd.read_csv(mashup_file)

    # Parse categories
    mashup_df['categories'] = mashup_df['categories'].apply(
        lambda x: x.split('|') if isinstance(x, str) and x else []
    )

    # Sample 5 diverse mashups
    sample_indices = [0, 100, 500, 1000, 5000]
    samples = []

    print("\n[3/3] Generating unified descriptions...")
    print("-" * 70)

    with torch.no_grad():
        for idx in sample_indices:
            row = mashup_df.iloc[idx]
            desc = str(row['description']) if pd.notna(row['description']) else ""
            cats = row['categories']
            name = str(row['name']) if pd.notna(row['name']) else f"Mashup_{idx}"

            print(f"\n{'='*70}")
            print(f"Sample #{idx}: {name}")
            print(f"{'='*70}")
            print(f"Original Description:")
            print(f"  {desc[:200]}...")
            print(f"Categories: {cats}")

            # Generate unified description using RPM
            print(f"\nGenerating unified description with use_generation=True...")

            # We need to manually call the generation to see the output
            # Let's look at the prompt
            from src.models.llm_semantic import RPM_PROMPT_TEMPLATE

            prompt = RPM_PROMPT_TEMPLATE.format(
                mashup_name=name,
                mashup_description=desc
            )

            # Encode and generate
            inputs = encoder.tokenizer(
                prompt,
                return_tensors='pt',
                max_length=encoder.max_length,
                truncation=True,
                padding=True
            ).to(device)

            # Generate
            outputs = encoder.llm.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,  # Deterministic
                do_sample=False,
                pad_token_id=encoder.tokenizer.pad_token_id
            )

            # Decode
            generated = encoder.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part (after the prompt)
            # Find where the actual generation starts
            if "Output:" in generated:
                generated_part = generated.split("Output:")[-1].strip()
            else:
                generated_part = generated[len(prompt):].strip()

            print(f"\nGenerated Unified Description:")
            print(f"  {generated_part[:300]}...")

            samples.append({
                'index': idx,
                'name': name,
                'original': desc[:100],
                'generated': generated_part[:200]
            })

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nChecking for template-like patterns...")

    # Check if generated descriptions start similarly
    starts = [s['generated'][:50] for s in samples]
    unique_starts = len(set(starts))

    print(f"\nFirst 50 chars of each generation:")
    for i, start in enumerate(starts):
        print(f"  #{sample_indices[i]}: {start}")

    print(f"\nUnique starting patterns: {unique_starts}/{len(samples)}")

    if unique_starts < len(samples):
        print(f"⚠️ WARNING: Generations have similar starting patterns!")
        print(f"   This suggests template-like outputs from TinyLlama")
    else:
        print(f"✓ Generations have diverse starting patterns")

    # Check average similarity
    print("\nChecking lexical overlap...")
    from collections import Counter

    word_sets = [set(s['generated'].lower().split()) for s in samples]
    common_words = set.intersection(*word_sets)

    print(f"Common words across all generations: {len(common_words)}")
    if len(common_words) > 20:
        print(f"⚠️ WARNING: Many common words ({len(common_words)}) across generations")
        print(f"   Common words: {list(common_words)[:20]}")

    print("\n" + "=" * 70)
    print("Check Complete")
    print("=" * 70)


if __name__ == "__main__":
    check_generation_diversity()
