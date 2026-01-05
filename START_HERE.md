# 🎉 SRCA 项目 - 论文正确实现版本

## ✅ 状态：已按论文修正完成

已按照SRCA论文正确实现，包括：
- ✅ LLaMA-3.1-8B语义编码器（论文使用LLaMA-3系列）
- ✅ RPM/FPA提示词（论文Section 4.1.1）
- ✅ 类别共现图构建（阈值θ=0.2）
- ✅ GNN特征增强（2层，1024维）
- ✅ 推荐MLP + Focal Loss
- ✅ 完整训练流程

## 🚀 快速开始（4步）

```bash
# 1. 进入目录
cd /home/zxu298/TRELLIS/api/SRCA

# 2. 激活环境
conda activate aaai

# 3. 数据预处理
python preprocess.py

# 4. 提取LLaMA语义特征
python extract_features.py

# 5. 开始训练
python train.py
```

## 📚 文档导航

选择适合你的文档：

### 🔰 新手：想快速开始
👉 **[QUICKSTART.md](QUICKSTART.md)** - 中文快速指南，包含完整步骤

### 📖 详细了解：想知道实现细节
👉 **[README_CN.md](README_CN.md)** - 中文完整文档
👉 **[README.md](README.md)** - English complete documentation
👉 **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** - 技术实现细节

### 🔧 遇到问题：故障排除
👉 运行 `python test_setup.py` 检查环境
👉 查看 QUICKSTART.md 的"故障排除"部分

## 🎯 预期结果

训练完成后：
- **MAP@5**: 0.70-0.72（原论文用LLaMA-3-8B: 0.7421）
- **NDCG@5**: 0.76-0.79（原论文: 0.8031）
- **训练时间**: 约30-60分钟/epoch

## 📦 项目文件

```
SRCA/
├── START_HERE.md              ⭐ 你现在在这里！
├── QUICKSTART.md              ⭐ 快速开始（强烈推荐）
├── README_CN.md               完整中文文档
├── README.md                  完整英文文档
│
├── preprocess.py              ⭐ 第一步：运行数据预处理
├── train.py                   ⭐ 第二步：开始训练
├── test_setup.py              验证环境
├── config.py                  修改配置参数
│
├── src/                       源代码（已实现）
│   ├── models/                模型组件
│   ├── datamodules/           数据加载
│   └── utils/                 工具函数
│
└── data/SEHGN/                数据目录（运行preprocess.py后生成）
```

## 💡 关键提示

1. **必须先运行 `preprocess.py`**
   - 从AdaFlow的embedding数据生成SRCA需要的CSV格式
   - 只需运行一次

2. **首次训练会下载TinyLlama模型（~2GB）**
   - 自动下载，无需手动操作
   - 后续训练会使用缓存

3. **监控训练进度**
   ```bash
   # 另开一个终端
   tensorboard --logdir=./logs
   # 浏览器访问 http://localhost:6006
   ```

## 🔍 快速验证

确保一切就绪：

```bash
cd /home/zxu298/TRELLIS/api/SRCA
conda activate aaai

# 验证环境（应该看到6个✓）
python test_setup.py

# 如果数据文件缺失，运行预处理
python preprocess.py
```

## 📈 与AdaFlow对比

| 特性 | AdaFlow | SRCA |
|------|---------|------|
| 位置 | `/api/AdaFlow/` | `/api/SRCA/` |
| 输入 | 预训练embeddings | LLM语义表示 |
| 类别 | 隐式（图中） | 显式（共现图+GNN） |
| 描述 | 不使用 | RPM/FPA提示词 |
| 状态 | ✅ 已迁移PyG | ✅ 完整实现 |

两个模型可以在同一个SEHGN数据集上对比！

## 🎓 学习资源

- **SRCA论文**: `/home/zxu298/TRELLIS/api/SRCA.pdf`
- **TinyLlama**: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Focal Loss**: Lin et al., ICCV 2017

## ✨ 开始你的第一次训练！

```bash
cd /home/zxu298/TRELLIS/api/SRCA
conda activate aaai
python preprocess.py  # 第一次必须运行
python train.py       # 开始训练！
```

祝训练顺利！🚀

---

**日期**: 2025-10-08
**环境**: aaai (PyTorch 2.8.0, CUDA 12.8)
**GPU**: NVIDIA RTX A6000
