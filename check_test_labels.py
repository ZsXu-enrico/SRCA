"""检查测试集的标签是否正确."""

import torch
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.datamodules.srca_datamodule import SRCADataModule

def check_test_labels():
    print("=" * 70)
    print("检查测试集标签")
    print("=" * 70)

    # 初始化datamodule
    datamodule = SRCADataModule(
        data_dir='./data/ProgrammableWeb/',
        batch_size=64,
        num_workers=0
    )

    datamodule.prepare_data()
    datamodule.setup()

    print(f"\n数据集大小：")
    print(f"  训练集: {len(datamodule.train_dataset)} mashups")
    print(f"  测试集: {len(datamodule.test_dataset)} mashups")

    # 获取测试dataloader
    test_loader = datamodule.test_dataloader()

    # 检查第一个batch
    print(f"\n检查第一个测试batch：")
    batch = next(iter(test_loader))

    print(f"  Batch size: {len(batch['mashup_indices'])}")
    print(f"  Mashup indices: {batch['mashup_indices'].tolist()[:10]}...")
    print(f"  Labels shape: {batch['labels'].shape}")

    # 统计正样本
    total_positives = batch['labels'].sum().item()
    positives_per_mashup = batch['labels'].sum(dim=1)

    print(f"\n  总正样本数: {total_positives}")
    print(f"  每个mashup的正样本数: {positives_per_mashup.tolist()}")

    if total_positives == 0:
        print(f"\n⚠️ 警告：测试batch中没有正样本！")

    # 检查整个测试集
    print(f"\n检查整个测试集：")
    total_test_positives = 0
    mashups_with_positives = 0

    for batch in test_loader:
        batch_positives = batch['labels'].sum().item()
        total_test_positives += batch_positives

        # 统计有正样本的mashup数量
        mashups_with_positives += (batch['labels'].sum(dim=1) > 0).sum().item()

    num_test_mashups = len(datamodule.test_dataset)
    num_apis = 1647

    print(f"  测试集mashup数: {num_test_mashups}")
    print(f"  总正样本数: {total_test_positives}")
    print(f"  有正样本的mashup数: {mashups_with_positives} / {num_test_mashups}")
    print(f"  正样本率: {total_test_positives / (num_test_mashups * num_apis) * 100:.4f}%")

    if total_test_positives == 0:
        print(f"\n❌ 错误：整个测试集没有正样本！")

        # 检查原始label矩阵
        print(f"\n检查原始label矩阵：")
        pair_file = './data/ProgrammableWeb/ma_pair.txt'
        pairs = pd.read_csv(pair_file, sep='\t', header=None, names=['mashup_id', 'api_id'])

        # 统计4000-4999范围内的pairs
        test_pairs = pairs[(pairs['mashup_id'] >= 4000) & (pairs['mashup_id'] < 5000)]

        print(f"  ma_pair.txt中4000-4999的pairs: {len(test_pairs)}")
        print(f"  前10个: ")
        print(test_pairs.head(10))

    else:
        print(f"\n✓ 测试集标签正常")

        # 显示几个有正样本的例子
        print(f"\n示例（前5个有正样本的mashup）：")
        count = 0
        for batch in test_loader:
            for i in range(len(batch['mashup_indices'])):
                if batch['labels'][i].sum() > 0:
                    mashup_idx = batch['mashup_indices'][i].item()
                    num_pos = batch['labels'][i].sum().item()
                    pos_apis = torch.where(batch['labels'][i] == 1)[0].tolist()[:5]

                    print(f"  Mashup {mashup_idx}: {num_pos} 个正样本, APIs: {pos_apis}...")

                    count += 1
                    if count >= 5:
                        break
            if count >= 5:
                break


if __name__ == "__main__":
    check_test_labels()
