"""Test script to verify LLM is actually generating unified descriptions."""

import os
import sys
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, os.path.dirname(__file__))

from config import SRCA_CONFIG
from src.models.llm_semantic import LLMSemanticEncoder

def test_generation():
    """Test if LLM generates unified descriptions."""

    print("=" * 70)
    print("Testing LLM Generation")
    print("=" * 70)

    # Initialize encoder
    print("\n[1/3] Initializing LLM Encoder...")
    encoder = LLMSemanticEncoder(
        model_name=SRCA_CONFIG['llm_model_name'],
        max_length=SRCA_CONFIG['llm_max_length'],
        semantic_dim=SRCA_CONFIG['semantic_dim'],
        freeze_llm=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    encoder.eval()
    print(f"✓ Encoder loaded on {device}")

    # Test Mashup encoding with generation
    print("\n[2/3] Testing Mashup Encoding (RPM) with use_generation=True...")
    mashup_desc = ["A social media dashboard that combines Twitter and Facebook feeds"]
    mashup_cats = [["Social", "Media"]]
    mashup_names = ["SocialHub"]

    with torch.no_grad():
        print("\n  Input description:")
        print(f"    {mashup_desc[0]}")

        # Test WITH generation
        print("\n  Encoding WITH generation (should generate unified description)...")
        features_with_gen = encoder.encode_mashups(
            mashup_desc,
            mashup_cats,
            mashup_names,
            use_generation=True
        )
        print(f"    Output shape: {features_with_gen.shape}")
        print(f"    Feature mean: {features_with_gen.mean().item():.4f}")
        print(f"    Feature std: {features_with_gen.std().item():.4f}")

        # Test WITHOUT generation for comparison
        print("\n  Encoding WITHOUT generation (direct hidden states)...")
        features_no_gen = encoder.encode_mashups(
            mashup_desc,
            mashup_cats,
            mashup_names,
            use_generation=False
        )
        print(f"    Output shape: {features_no_gen.shape}")
        print(f"    Feature mean: {features_no_gen.mean().item():.4f}")
        print(f"    Feature std: {features_no_gen.std().item():.4f}")

        # Compare features
        diff = torch.abs(features_with_gen - features_no_gen).mean().item()
        print(f"\n  Feature difference (with_gen vs no_gen): {diff:.4f}")
        if diff > 0.1:
            print("  ✓ Features are DIFFERENT - generation is working!")
        else:
            print("  ✗ Features are SIMILAR - generation might not be working!")

    # Test API encoding with generation
    print("\n[3/3] Testing API Encoding (FPA) with use_generation=True...")
    api_names = ["Twitter API"]
    api_descs = ["Access tweets, timelines, and user information from Twitter"]
    api_cats = [["Social"]]

    with torch.no_grad():
        print("\n  Input description:")
        print(f"    {api_descs[0]}")

        # Test WITH generation
        print("\n  Encoding WITH generation (should generate unified description)...")
        features_with_gen = encoder.encode_apis(
            api_names,
            api_descs,
            api_cats,
            use_generation=True
        )
        print(f"    Output shape: {features_with_gen.shape}")
        print(f"    Feature mean: {features_with_gen.mean().item():.4f}")
        print(f"    Feature std: {features_with_gen.std().item():.4f}")

        # Test WITHOUT generation for comparison
        print("\n  Encoding WITHOUT generation (direct hidden states)...")
        features_no_gen = encoder.encode_apis(
            api_names,
            api_descs,
            api_cats,
            use_generation=False
        )
        print(f"    Output shape: {features_no_gen.shape}")
        print(f"    Feature mean: {features_no_gen.mean().item():.4f}")
        print(f"    Feature std: {features_no_gen.std().item():.4f}")

        # Compare features
        diff = torch.abs(features_with_gen - features_no_gen).mean().item()
        print(f"\n  Feature difference (with_gen vs no_gen): {diff:.4f}")
        if diff > 0.1:
            print("  ✓ Features are DIFFERENT - generation is working!")
        else:
            print("  ✗ Features are SIMILAR - generation might not be working!")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    print("\nKey Findings:")
    print("- If feature differences > 0.1: Generation is working correctly ✓")
    print("- If feature differences < 0.1: Generation might not be working ✗")
    print("\nNote: With generation=True, LLM should generate unified descriptions")
    print("      before extracting features, resulting in different features.")


if __name__ == "__main__":
    test_generation()
