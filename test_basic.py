"""Basic test to diagnose issues."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.insert(0, os.path.dirname(__file__))

print("=" * 50)
print("Step 1: Testing imports...")
print("=" * 50)

try:
    from config import SRCA_CONFIG
    print("✓ Config imported")

    from src.datamodules.srca_datamodule import SRCADataModule
    print("✓ DataModule imported")

    from src.models.srca_model import SRCALightningModule
    print("✓ Model imported")

except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("Step 2: Testing DataModule...")
print("=" * 50)

try:
    datamodule = SRCADataModule(
        data_dir=SRCA_CONFIG['data_dir'],
        batch_size=2,
        num_workers=0
    )
    datamodule.prepare_data()
    datamodule.setup()
    print(f"✓ Train: {len(datamodule.train_dataset)} samples")
    print(f"✓ Val: {len(datamodule.val_dataset)} samples")

except Exception as e:
    print(f"❌ DataModule error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("Step 3: Testing Model initialization...")
print("=" * 50)

try:
    import torch
    model = SRCALightningModule(
        num_apis=SRCA_CONFIG['num_apis'],
        semantic_dim=SRCA_CONFIG['semantic_dim'],
        gnn_num_layers=SRCA_CONFIG['gnn_num_layers'],
        mlp_hidden_dims=SRCA_CONFIG['mlp_hidden_dims'],
        mlp_dropout=SRCA_CONFIG['mlp_dropout'],
        llm_model_name=SRCA_CONFIG['llm_model_name'],
        llm_max_length=SRCA_CONFIG['llm_max_length'],
        freeze_llm=True,
        focal_alpha=SRCA_CONFIG['focal_alpha'],
        focal_gamma=SRCA_CONFIG['focal_gamma'],
        learning_rate=SRCA_CONFIG['learning_rate'],
        weight_decay=SRCA_CONFIG['weight_decay'],
        mashup_api_graph_edge_index=None,
        mashup_api_graph_edge_weight=None,
        eval_k_values=[1, 3, 5, 10]
    )
    # Move model to CUDA
    if torch.cuda.is_available():
        model = model.cuda()
        print("✓ Model moved to CUDA")

    print("✓ Model initialized (this will download TinyLlama if not cached)")

except Exception as e:
    print(f"❌ Model error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("Step 4: Testing forward pass...")
print("=" * 50)

try:
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    # Extract features
    mashup_features = model.llm_encoder.encode_mashups(
        batch['descriptions'],
        batch['categories'],
        use_generation=False
    )
    print(f"✓ Extracted mashup features: {mashup_features.shape}")

    # Dummy API features (on same device as model)
    device = mashup_features.device
    api_features = torch.zeros(SRCA_CONFIG['num_apis'], mashup_features.size(1), device=device)

    # Forward pass
    logits = model(mashup_features, api_features)
    print(f"✓ Forward pass successful: {logits.shape}")

    # Compute loss
    labels = batch['labels'].to(device)
    loss = model.criterion(logits, labels)
    print(f"✓ Loss computed: {loss.item():.4f}")

except Exception as e:
    print(f"❌ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("✅ All tests passed!")
print("=" * 50)
print("\nYou can now run the full training with:")
print("  conda run -n aaai python train_simplified.py")
