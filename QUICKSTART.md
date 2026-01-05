# SRCA 快速开始指南 🚀

## 目录结构

```
/home/zxu298/TRELLIS/api/
├── AdaFlow/           # AdaFlow模型（已完成迁移）
│   └── data/SEHGN/    # 原始SEHGN数据
└── SRCA/              # SRCA模型（新实现）
    ├── data/SEHGN/    # 预处理后的数据
    ├── src/           # 源代码
    └── train.py       # 训练脚本
```

## 快速开始（3步）

### 1. 数据预处理

```bash
conda activate aaai
cd /home/zxu298/TRELLIS/api/SRCA
python preprocess.py
```

**输出**：
- `data/SEHGN/mashup.csv` - 8217个mashup，包含描述和类别
- `data/SEHGN/api.csv` - 1647个API，包含描述和类别
- `data/SEHGN/ma_pair.txt` - 36936个交互对

**预处理做了什么**：
- 从AdaFlow的embedding数据生成服务描述
- 使用K-means聚类分配类别（50个类别）
- 生成真实的mashup-API交互对（平均每个mashup调用4.5个API）
- 创建SRCA训练所需的CSV格式数据

### 2. 验证安装

```bash
python test_setup.py
```

应该看到所有✓通过：
- ✓ 所有模块导入成功
- ✓ CUDA可用（NVIDIA RTX A6000）
- ✓ 数据文件存在
- ✓ 组件测试通过

### 3. 开始训练

```bash
python train.py
```

**首次运行会下载TinyLlama模型（~2GB）**

## 训练配置

默认配置在 `config.py`:

```python
SRCA_CONFIG = {
    # LLM
    'llm_model_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'semantic_dim': 768,

    # GNN
    'gnn_num_layers': 2,
    'gnn_hidden_dim': 256,
    'gnn_theta': 0.2,

    # 训练
    'batch_size': 32,
    'learning_rate': 1e-4,
    'max_epochs': 100,
    'patience': 10,
}
```

## 查看训练进度

```bash
# 在另一个终端启动TensorBoard
tensorboard --logdir=./logs

# 浏览器打开 http://localhost:6006
```

## 预期结果

训练完成后会生成：

1. **检查点** - `checkpoints/srca-{epoch}-{val_P@5}.ckpt`
2. **日志** - `logs/srca_tinyllama/`
3. **测试结果** - 自动在最佳checkpoint上测试

**预期性能**（TinyLlama vs 论文中的LLaMA-3-8B）：
- MAP@5: ~0.70-0.72（论文：0.7421）
- NDCG@5: ~0.76-0.79（论文：0.8031）
- P@5: ~0.60-0.65

## 文件说明

```
SRCA/
├── preprocess.py              # 数据预处理脚本 ⭐
├── train.py                   # 训练主脚本
├── test_setup.py              # 安装验证
├── config.py                  # 所有配置参数
├── requirements.txt           # Python依赖
│
├── src/
│   ├── models/
│   │   ├── llm_semantic.py      # TinyLlama编码器
│   │   ├── gnn_augmentation.py  # GNN特征增强
│   │   ├── recommendation_mlp.py # 推荐MLP
│   │   ├── focal_loss.py        # Focal Loss
│   │   └── srca_model.py        # 完整SRCA模型
│   │
│   ├── datamodules/
│   │   └── srca_datamodule.py   # 数据加载器
│   │
│   └── utils/
│       ├── prompts.py           # RPM/FPA提示词模板
│       └── category_graph.py    # 类别共现图
│
└── data/SEHGN/                # 预处理后的数据
    ├── mashup.csv
    ├── api.csv
    └── ma_pair.txt
```

## 关键技术点

### 1. LLM语义表示
- **模型**：TinyLlama-1.1B-Chat（轻量级，但保持语义理解能力）
- **RPM提示词**：统一mashup描述格式
- **FPA提示词**：提取API关键特征
- **特征提取**：从最后一层hidden states提取768维语义向量

### 2. 类别共现图
- **节点**：50个服务类别（Social, Tools, Media等）
- **边**：基于共现频率加权
- **作用**：捕获类别间的关系（如Social常与Media共现）

### 3. GNN特征增强
- **架构**：2层GCN
- **公式**：`h_final = 0.2 * h_aug + 0.8 * h_cat`
- **作用**：用图结构信息增强类别特征

### 4. Focal Loss
- **参数**：α=0.25, γ=2.0
- **作用**：解决API推荐中的类别不平衡问题
- **效果**：聚焦难分类样本，提升长尾API推荐

## 测试单个组件

```bash
# 测试LLM编码器
python -c "from src.models.llm_semantic import test_llm_encoder; test_llm_encoder()"

# 测试类别图
python src/utils/category_graph.py

# 测试GNN
python src/models/gnn_augmentation.py

# 测试MLP
python src/models/recommendation_mlp.py

# 测试Focal Loss
python src/models/focal_loss.py
```

## 自定义配置

修改 `config.py` 来调整超参数：

```python
# 如果GPU内存不足
'batch_size': 16,              # 默认32
'llm_load_in_8bit': True,      # 使用8bit量化

# 调整学习率
'learning_rate': 5e-5,         # 默认1e-4

# 改变GNN深度
'gnn_num_layers': 3,           # 默认2
'gnn_theta': 0.3,              # 默认0.2

# Early stopping
'patience': 15,                # 默认10
```

## 故障排除

### 1. CUDA内存不足
```python
# config.py
'batch_size': 16,  # 减小batch size
'llm_load_in_8bit': True,  # 启用8bit量化
```

### 2. 训练太慢
```python
# 已启用mixed precision (FP16)
# 已启用Tensor Cores
# 可以减少max_epochs进行快速测试
```

### 3. 数据文件丢失
```bash
# 重新运行预处理
python preprocess.py
```

## 与AdaFlow对比

| 特性 | AdaFlow | SRCA |
|------|---------|------|
| 输入 | 预训练embeddings | LLM生成语义表示 |
| 类别 | 隐式（图结构中） | 显式（共现图+GNN） |
| 损失 | 标准BCE | Focal Loss |
| 描述 | 不使用 | 通过提示词生成 |

## 性能监控

训练过程中会记录：
- **训练指标**：loss
- **验证指标**：P@K, R@K, NDCG@K, MAP@K（K=1,3,5,10）
- **最佳模型**：基于val/P@5

## 联系与反馈

有问题？检查：
1. `README.md` - 详细文档
2. `IMPLEMENTATION_COMPLETE.md` - 实现细节
3. `QUICKSTART.md` - 本文件

---

**创建日期**：2025-10-08
**环境**：aaai (PyTorch 2.8.0, PyG 2.6.1)
**GPU**：NVIDIA RTX A6000
**状态**：✅ 完全可运行
