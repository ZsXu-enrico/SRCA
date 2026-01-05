# SRCA - 基于TinyLlama的API推荐系统

完整实现了SRCA论文中的模型架构，使用TinyLlama-1.1B替代原论文中的LLaMA-3-8B以提高效率。

## 📋 目录

- [快速开始](#快速开始)
- [模型架构](#模型架构)
- [文件结构](#文件结构)
- [详细文档](#详细文档)
- [性能对比](#性能对比)

## 🚀 快速开始

### 1. 数据预处理（必须）

```bash
conda activate aaai
cd /home/zxu298/TRELLIS/api/SRCA
python preprocess.py
```

这会从AdaFlow的embedding数据生成SRCA需要的CSV格式数据：
- ✓ 8217个mashup（包含AI生成的描述和类别）
- ✓ 1647个API（包含AI生成的描述和类别）
- ✓ 36936个mashup-API交互对

### 2. 验证环境

```bash
python test_setup.py
```

应该看到所有6项测试通过 ✓

### 3. 开始训练

```bash
python train.py
```

**注意**：首次运行会自动下载TinyLlama模型（约2GB）

## 🏗️ 模型架构

```
输入: Mashup描述
   ↓
┌─────────────────────────────────────┐
│ LLM语义编码器 (TinyLlama-1.1B)      │
│ - RPM提示词：统一描述格式            │
│ - 提取768维语义特征                  │
└─────────────────────────────────────┘
   ↓ 语义特征 (768-dim)
   ↓
┌─────────────────────────────────────┐
│ 类别共现图 + GNN增强                 │
│ - 50个类别节点                       │
│ - 基于共现频率的边权重                │
│ - 2层GCN特征传播                     │
│ - θ=0.2加权融合                      │
└─────────────────────────────────────┘
   ↓ 增强特征
   ↓
┌─────────────────────────────────────┐
│ 推荐MLP                              │
│ 768 → 512 → 256 → 128 → 1647       │
│ Focal Loss (α=0.25, γ=2.0)         │
└─────────────────────────────────────┘
   ↓
输出: Top-K API推荐
```

## 📁 文件结构

```
SRCA/
├── 📄 核心脚本
│   ├── preprocess.py              ⭐ 数据预处理（必须先运行）
│   ├── train.py                   ⭐ 训练主脚本
│   ├── test_setup.py              验证环境
│   └── config.py                  所有配置参数
│
├── 📚 文档
│   ├── QUICKSTART.md              快速开始指南（中文）
│   ├── README.md                  详细说明（英文）
│   ├── README_CN.md               本文件
│   └── IMPLEMENTATION_COMPLETE.md 实现细节
│
├── 🔧 源代码
│   └── src/
│       ├── models/
│       │   ├── llm_semantic.py      TinyLlama语义编码器
│       │   ├── gnn_augmentation.py  GNN特征增强
│       │   ├── recommendation_mlp.py 推荐MLP
│       │   ├── focal_loss.py        Focal Loss实现
│       │   └── srca_model.py        完整SRCA模型
│       ├── datamodules/
│       │   └── srca_datamodule.py   数据加载器
│       └── utils/
│           ├── prompts.py           提示词模板（RPM/FPA）
│           └── category_graph.py    类别共现图
│
└── 💾 数据（运行preprocess.py后生成）
    └── data/SEHGN/
        ├── mashup.csv            Mashup描述和类别
        ├── api.csv               API描述和类别
        └── ma_pair.txt           交互对
```

## 📖 详细文档

1. **[QUICKSTART.md](QUICKSTART.md)** - 快速开始，包含：
   - 完整的3步启动流程
   - 配置调整说明
   - 故障排除指南

2. **[README.md](README.md)** - 完整的英文文档

3. **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** - 技术实现细节

## 🎯 核心特性

### 1. LLM语义表示
- **模型**: TinyLlama-1.1B-Chat（2048隐藏层 → 投影到768维）
- **提示词工程**:
  - RPM (Representation Prompt for Mashups): 统一mashup描述
  - FPA (Feature Prompt for APIs): 提取API关键特征
- **特征提取**: 从最后一层隐藏状态平均池化

### 2. 类别共现图
- **节点**: 50个真实服务类别（Social, Tools, Media等）
- **边**: 基于K-means聚类自动分配
- **权重**: 共现频率（阈值0.01）

### 3. GNN特征增强
- **架构**: 2层GCN（256隐藏维度）
- **融合公式**: `h = 0.2 * h_GNN + 0.8 * h_cat`
- **效果**: 利用类别关系增强语义特征

### 4. Focal Loss
- **参数**: α=0.25, γ=2.0
- **作用**: 解决API推荐中的长尾问题
- **效果**: 提升难样本的学习效果

## 📊 性能对比

### 与原论文对比

| 模型组件 | 原论文SRCA | 本实现 | 说明 |
|---------|-----------|--------|------|
| LLM | LLaMA-3-8B | TinyLlama-1.1B | 轻量化，保持语义能力 |
| 语义维度 | 未说明 | 768 | TinyLlama隐藏层投影 |
| GNN层数 | 2 | 2 | ✓ 完全一致 |
| GNN θ | 0.2 | 0.2 | ✓ 完全一致 |
| Focal α | 0.25 | 0.25 | ✓ 完全一致 |
| Focal γ | 2.0 | 2.0 | ✓ 完全一致 |

### 预期性能

基于TinyLlama（vs 原论文LLaMA-3-8B）：

| 指标 | 原论文 | 预期 | 差异 |
|-----|--------|------|------|
| MAP@5 | 0.7421 | 0.70-0.72 | -3% |
| NDCG@5 | 0.8031 | 0.76-0.79 | -3% |
| P@5 | ~0.65 | 0.60-0.65 | -5% |

**训练时间**: 约30-60分钟/epoch（NVIDIA RTX A6000）

## ⚙️ 配置调整

编辑 `config.py`:

```python
# GPU内存不足时
SRCA_CONFIG = {
    'batch_size': 16,           # 默认32
    'llm_load_in_8bit': True,   # 使用8bit量化
}

# 调整学习率
SRCA_CONFIG = {
    'learning_rate': 5e-5,      # 默认1e-4
}

# 调整GNN结构
SRCA_CONFIG = {
    'gnn_num_layers': 3,        # 默认2
    'gnn_theta': 0.3,           # 默认0.2
}
```

## 🔍 监控训练

### TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir=./logs

# 浏览器访问 http://localhost:6006
```

### 指标说明

训练过程监控：
- `train/loss`: 训练损失（Focal Loss）
- `val/P@K`: 验证集Precision@K（K=1,3,5,10）
- `val/R@K`: 验证集Recall@K
- `val/NDCG@K`: 验证集NDCG@K
- `val/MAP@K`: 验证集MAP@K

**最佳模型**: 基于 `val/P@5` 保存

## 🧪 测试组件

每个组件都有独立的测试函数：

```bash
# 测试LLM编码器
python -c "from src.models.llm_semantic import test_llm_encoder; test_llm_encoder()"

# 测试类别图构建
python src/utils/category_graph.py

# 测试GNN增强
python src/models/gnn_augmentation.py

# 测试推荐MLP
python src/models/recommendation_mlp.py

# 测试Focal Loss
python src/models/focal_loss.py
```

## 🐛 故障排除

### 1. CUDA内存不足 (OOM)

```python
# config.py
SRCA_CONFIG = {
    'batch_size': 8,            # 从32减小
    'llm_load_in_8bit': True,   # 启用8bit量化
}
```

### 2. 数据文件缺失

```bash
# 重新运行预处理
cd /home/zxu298/TRELLIS/api/SRCA
python preprocess.py
```

### 3. transformers包缺失

```bash
conda activate aaai
pip install transformers accelerate sentencepiece
```

### 4. 训练太慢

- ✓ 已启用混合精度训练（FP16）
- ✓ 已启用Tensor Cores
- 可以减少 `max_epochs` 进行快速测试

## 📦 依赖环境

- **Python**: 3.8+
- **PyTorch**: 2.8.0 + CUDA 12.8
- **PyTorch Geometric**: 2.6.1
- **Transformers**: 4.36.0+
- **PyTorch Lightning**: 2.0+

所有依赖已在aaai环境中安装 ✓

## 🔗 与AdaFlow的关系

```
/home/zxu298/TRELLIS/api/
├── AdaFlow/              使用预训练embedding
│   └── data/SEHGN/       原始embedding数据
└── SRCA/                 使用LLM生成语义表示
    └── data/SEHGN/       从AdaFlow数据预处理得到
```

- **数据共享**: SRCA的预处理脚本从AdaFlow的embedding数据生成描述
- **评估对比**: 可以在相同的SEHGN数据集上比较两个模型
- **互补性**: AdaFlow侧重图结构，SRCA侧重语义理解

## 🎓 论文引用

```bibtex
@article{srca2024,
  title={Semantic Representation with Category Augmentation for Web API Recommendation},
  year={2024}
}
```

## ✅ 当前状态

- ✅ 所有组件完整实现
- ✅ 数据预处理脚本就绪
- ✅ 训练流程验证通过
- ✅ 环境完全配置
- ✅ 文档完善

**可以直接开始训练！**

## 📞 需要帮助？

1. 查看 [QUICKSTART.md](QUICKSTART.md) 获取详细步骤
2. 查看 [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) 了解技术细节
3. 运行 `python test_setup.py` 验证环境

---

**实现日期**: 2025-10-08
**环境**: aaai (PyTorch 2.8.0, PyG 2.6.1, CUDA 12.8)
**GPU**: NVIDIA RTX A6000
**状态**: ✅ 完全可运行
