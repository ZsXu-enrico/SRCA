# 🚀 SRCA 新服务器部署指南

本仓库已包含所有预处理好的特征文件，可以在新服务器上**直接训练**，无需重新提取特征！

## ✅ 已包含的预处理文件

```bash
data/
├── semantic_features.pt    # 20MB - LLaMA语义特征（已提取）
├── category_graph.pt       # 37MB - 类别共现图（已构建）
├── mashup.csv             # 1.3MB - Mashup数据
├── api.csv                # 590KB - API数据
└── ma_pair.txt            # 152KB - 交互对数据
```

这些文件已经包含在仓库中，**不需要**重新运行 `extract_features.py`！

## 📦 新服务器部署步骤

### 1. 克隆仓库

```bash
git clone https://github.com/ZsXu-enrico/SRCA.git
cd SRCA
```

### 2. 创建环境

```bash
conda create -n srca python=3.8
conda activate srca
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置路径（如需要）

编辑 `config.py`，修改数据路径（如果不同）：

```python
SRCA_CONFIG = {
    'data_dir': './data',  # 确保指向正确的数据目录
    # ... 其他配置
}
```

### 5. 直接开始训练！🎉

```bash
python train.py
```

**就这么简单！** 因为所有特征已经预提取，训练会立即开始。

## 📊 训练时的GPU内存监控

训练脚本已经添加了GPU内存监控功能，测试阶段会自动显示：

```
======================================================================
GPU Memory Usage (Before Testing)
======================================================================
  Allocated:        XXX.XX MB (  X.XX GB)
  Reserved:         XXX.XX MB (  X.XX GB)
  Max Allocated:    XXX.XX MB (  X.XX GB)
======================================================================
```

## 🔧 可选：调整配置

如果GPU内存不足，可以在 `config.py` 中调整：

```python
SRCA_CONFIG = {
    'batch_size': 16,           # 默认32，减小以节省内存
    'use_8bit': True,           # 启用8bit量化（仅用于LLM）
    # ...
}
```

## 📈 预期结果

- **MAP@5**: 0.70-0.72
- **NDCG@5**: 0.76-0.79
- **训练时间**: 约30-60分钟/epoch（取决于GPU）

## 🔍 查看训练日志

```bash
tensorboard --logdir=./logs
# 浏览器访问 http://localhost:6006
```

## 📁 文件说明

### 核心训练文件

- `train.py` - 主训练脚本（带GPU内存监控）
- `config.py` - 所有配置参数
- `requirements.txt` - Python依赖

### 源代码

```
src/
├── models/
│   ├── srca_model.py          # SRCA主模型
│   ├── llm_semantic.py        # LLM语义编码器
│   ├── gnn_augmentation.py    # GNN特征增强
│   └── ...
├── datamodules/
│   └── srca_datamodule.py     # 数据加载
└── utils/
    └── ...
```

### 可选脚本

- `extract_features.py` - LLM特征提取（**已运行，不需要再执行**）
- `preprocess.py` - 数据预处理（**已运行，不需要再执行**）

## ❓ 故障排除

### 1. 找不到数据文件

确保 `data/` 目录存在且包含所有文件：

```bash
ls -lh data/
# 应该看到：
# semantic_features.pt (20MB)
# category_graph.pt (37MB)
# mashup.csv, api.csv, ma_pair.txt
```

### 2. CUDA内存不足

减小 batch size 或启用量化：

```python
# config.py
SRCA_CONFIG = {
    'batch_size': 8,     # 从32减到8
    'use_8bit': True,    # 启用8bit量化
}
```

### 3. 缺少transformers包

```bash
pip install transformers accelerate sentencepiece
```

## 🎯 与旧流程的对比

### ❌ 旧流程（需要重新提取特征）

```bash
git clone ...
cd SRCA
pip install -r requirements.txt
python preprocess.py          # 步骤1
python extract_features.py    # 步骤2 - 很耗时！需要下载LLaMA模型
python train.py               # 步骤3
```

### ✅ 新流程（特征已包含）

```bash
git clone ...
cd SRCA
pip install -r requirements.txt
python train.py               # 直接训练！
```

**节省时间**：不需要重新下载LLaMA模型（~2GB），不需要重新提取特征（耗时数小时）

## 📝 注意事项

1. **token_features.pt (2.9GB)** 未上传 - 这是实验性文件，当前训练不需要
2. **checkpoints/** 和 **logs/** 已排除 - 这些会在训练时自动生成
3. **所有源代码和预处理数据已包含** - 可以立即开始训练

## 🎓 文档

- `START_HERE.md` - 快速开始指南
- `README_CN.md` - 完整中文文档
- `README.md` - 完整英文文档

## 💡 提示

如果你想从头开始重新提取特征（不推荐，除非修改了数据）：

```bash
# 删除预提取的特征
rm data/semantic_features.pt data/category_graph.pt

# 重新提取（会下载LLaMA模型）
python extract_features.py

# 然后训练
python train.py
```

但通常情况下，**直接使用预提取的特征更快更方便**！

---

**部署完成！** 祝训练顺利！🚀