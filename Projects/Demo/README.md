# VMC Demo Project

**Variational Memory Compression (VMC)** 演示项目

一个展示变分记忆压缩架构的完整demo，包含详细的可视化功能，专为MacBook Air M1 16GB内存优化。

## 🏗️ 项目架构

```
VMC = V (Variational) + M (Memory) + C (Controller)

┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  V Component │───▶│  M Component │───▶│ C Component │
│  (Variational)│    │   (Memory)   │    │ (Controller)│
└─────────────┘    └──────────────┘    └─────────────┘
```

### 核心组件

- **V组件 (Variational Encoder)**: 将输入编码为变分潜在表示
- **M组件 (Memory Module)**: 使用混合高斯分布的记忆机制
- **C组件 (Controller)**: 融合变分和记忆信息进行决策

## 📁 项目结构

```
Demo/
├── config.py              # 配置管理
├── dataset.py             # MNIST数据集处理
├── vmc_model.py           # VMC模型架构
├── visualizer.py          # 详细可视化模块
├── trainer.py             # 训练逻辑
├── main.py                # 主入口文件
├── README.md              # 说明文档
├── requirements.txt       # 依赖包
└── results/               # 可视化结果
    ├── variational_analysis_epoch_X.png
    ├── memory_analysis_epoch_X.png
    ├── controller_analysis_epoch_X.png
    └── complete_pipeline_epoch_X.png
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 确保PyTorch支持MPS (Apple Silicon)
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### 2. 运行方式

#### 交互式运行（推荐）
```bash
python main.py
```

#### 命令行模式
```bash
# 演示模式 (5分钟快速演示)
python main.py --mode demo

# 训练模式
python main.py --mode train

# 快速训练
python main.py --quick

# 可视化模式
python main.py --mode visualize

# 测试模式
python main.py --mode test

# 查看配置
python main.py --config
```

#### 直接模块运行
```bash
# 测试各个模块
python dataset.py      # 测试数据集
python vmc_model.py    # 测试模型
python trainer.py      # 开始训练
```

## 🎨 可视化功能

VMC Demo的核心特色是详细的可视化功能，展示每个组件的工作过程：

### V组件可视化
- 变分编码的t-SNE/PCA投影
- 潜在空间均值和方差分布
- KL散度分析
- 编码质量评估

### M组件可视化
- 注意力权重热图
- 混合高斯分布可视化
- 记忆槽激活模式
- 查询-检索相似性分析

### C组件可视化
- 门控权重分布
- 预测置信度分析
- 混淆矩阵
- 决策边界可视化

### 完整流水线
- 端到端的数据流可视化
- 各组件输出的综合展示

## ⚙️ 配置说明

### 默认配置（适合MacBook Air M1 16GB）

```python
# 数据配置
batch_size = 32          # 批次大小
seq_length = 50          # 序列长度  
input_dim = 784          # 输入维度 (28×28)

# V组件配置
variational_dim = 64     # 变分潜在空间维度
encoder_hidden_dims = [512, 256, 128]

# M组件配置  
memory_size = 16         # 记忆槽数量
memory_dim = 32          # 记忆维度
num_gaussians = 8        # 混合高斯组件数

# C组件配置
controller_hidden_dim = 128
controller_output_dim = 10  # MNIST 10分类

# 训练配置
epochs = 30              # 训练轮数
learning_rate = 1e-3     # 学习率
```

### 配置模式

- **快速模式**: 10 epochs, 适合演示
- **标准模式**: 30 epochs, 平衡效果和时间  
- **完整模式**: 50 epochs, 最佳效果
- **自定义模式**: 用户自定义参数

## 📊 性能基准

在MacBook Air M1 16GB上的性能表现：

| 配置 | 训练时间 | 内存使用 | MNIST准确率 |
|------|----------|----------|-------------|
| 快速模式 | ~10分钟 | ~4GB | ~95% |
| 标准模式 | ~30分钟 | ~6GB | ~97% |  
| 完整模式 | ~50分钟 | ~8GB | ~98% |

## 🎯 使用场景

### 1. 学习和研究
- 理解变分自编码器原理
- 学习记忆机制在深度学习中的应用
- 研究不同组件的协同工作

### 2. 快速原型验证
- 测试VMC架构在其他数据集上的效果
- 验证记忆机制的有效性
- 比较不同配置的性能

### 3. 教学演示
- 课程演示深度学习概念
- 可视化机器学习训练过程
- 展示模块化设计的优势

## 🔧 故障排除

### 常见问题

1. **MPS不可用**
   ```bash
   # 检查MPS支持
   python -c "import torch; print(torch.backends.mps.is_available())"
   # 如果不支持，会自动使用CPU
   ```

2. **内存不足**
   - 减小 `batch_size` (32 → 16)
   - 减小 `memory_size` (16 → 8)
   - 减小 `variational_dim` (64 → 32)

3. **训练速度慢**
   - 使用快速模式 (`--quick`)
   - 减少 `epochs` 数量
   - 关闭部分可视化

4. **可视化失败**
   - 确保安装了matplotlib和seaborn
   - 检查显示设置 (如果使用SSH)
   - 减少可视化的数据量

### 性能优化

```python
# 针对不同硬件的优化建议

# 8GB内存设备
VMCConfig.batch_size = 16
VMCConfig.memory_size = 8
VMCConfig.variational_dim = 32

# 16GB内存设备 (推荐)
VMCConfig.batch_size = 32  
VMCConfig.memory_size = 16
VMCConfig.variational_dim = 64

# 32GB+内存设备
VMCConfig.batch_size = 64
VMCConfig.memory_size = 32
VMCConfig.variational_dim = 128
```

## 📈 扩展功能

### 添加新数据集
```python
# 在dataset.py中添加新的数据集类
class CustomDataset(Dataset):
    def __init__(self, ...):
        # 自定义数据集实现
        pass
```

### 修改模型架构
```python
# 在vmc_model.py中扩展组件
class EnhancedMemoryModule(MemoryModule):
    def __init__(self, ...):
        # 增强的记忆模块
        pass
```

### 自定义可视化
```python
# 在visualizer.py中添加新的可视化方法
def custom_visualization(self, ...):
    # 自定义可视化逻辑
    pass
```

## 📝 更新日志

- **v1.0.0** - 初始版本，基础VMC实现
- **v1.1.0** - 添加详细可视化功能
- **v1.2.0** - MacBook Air M1优化
- **v1.3.0** - 交互式界面和演示模式

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境
```bash
git clone <repo>
cd Demo
pip install -r requirements.txt
python main.py --mode test  # 运行测试
```

## 📄 许可证

MIT License

## 🙋 常见问题 (FAQ)

**Q: VMC和普通VAE有什么区别？**
A: VMC在VAE基础上添加了记忆机制(M组件)和控制器(C组件)，能够存储和检索有用的模式，提升表示学习能力。

**Q: 为什么使用混合高斯分布？**
A: 混合高斯分布比单一高斯分布更灵活，能够建模复杂的多模态分布，更好地表示记忆中的多样性。

**Q: 如何调试训练过程？**
A: 使用可视化功能实时观察各组件的输出，检查KL散度、注意力权重、门控权重等指标。

**Q: 可以用于其他任务吗？**
A: 可以！只需修改输出维度和损失函数，VMC架构可以应用于回归、序列建模等多种任务。