"""简单的训练脚本."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from simple_datamodule import SimpleDataModule
from simple_model import SimpleModel

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("=" * 70)
    print("Simple Baseline Training")
    print("=" * 70)

    # 设置seed
    pl.seed_everything(42, workers=True)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")

    # Step 1: 加载数据
    print("\n" + "=" * 70)
    print("Step 1: Loading Data")
    print("=" * 70)

    datamodule = SimpleDataModule(
        data_dir='../data/',
        batch_size=64,
        num_workers=4
    )

    datamodule.setup()

    print(f"✓ Train: {len(datamodule.train_dataset)} samples")
    print(f"✓ Test: {len(datamodule.test_dataset)} samples")

    # Step 2: 加载token features
    print("\n" + "=" * 70)
    print("Step 2: Loading Token Features")
    print("=" * 70)

    token_file = './data/ProgrammableWeb/token_features.pt'
    if not os.path.exists(token_file):
        print(f"❌ Token features not found: {token_file}")
        print("Please run: python extract_features_token_level.py")
        return

    print(f"Loading from {token_file}...")
    token_data = torch.load(token_file, map_location='cpu', weights_only=False)

    # 只用训练集的4000个
    num_train_mashups = 4000
    train_mashup_tokens = token_data['mashup_token_features'][:num_train_mashups]

    print(f"✓ Train mashup tokens: {train_mashup_tokens.shape}")

    # Step 3: 初始化模型
    print("\n" + "=" * 70)
    print("Step 3: Initializing Model")
    print("=" * 70)

    model = SimpleModel(
        num_apis=1647,
        seq_len=50,
        hidden_dim=1536,
        learning_rate=1e-4,
        train_mashup_token_features=train_mashup_tokens
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Step 4: 设置训练
    print("\n" + "=" * 70)
    print("Step 4: Setting up Training")
    print("=" * 70)

    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='simple-baseline-last',
        save_top_k=0,
        save_last=True,
        verbose=False
    )

    logger_tb = TensorBoardLogger(
        save_dir='./logs',
        name='simple_baseline'
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=logger_tb,
        log_every_n_steps=10,
        deterministic=True,
        precision='16-mixed' if torch.cuda.is_available() else '32',
        limit_val_batches=0,
    )

    print("✓ Trainer configured")
    print(f"  Max epochs: 10")
    print(f"  Batch size: 64")
    print(f"  Learning rate: 1e-4")

    # Step 5: 训练
    print("\n" + "=" * 70)
    print("Step 5: Training")
    print("=" * 70)

    try:
        trainer.fit(model, datamodule=datamodule)

        print("\n" + "=" * 70)
        print("Training Completed!")
        print("=" * 70)
        print(f"Last checkpoint: {checkpoint_callback.last_model_path}")

        # Step 6: 测试
        print("\n" + "=" * 70)
        print("Step 6: Testing")
        print("=" * 70)

        trainer.test(model=model, datamodule=datamodule, ckpt_path='last')

        print("\n" + "=" * 70)
        print("Done!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
