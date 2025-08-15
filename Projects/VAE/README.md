# VAE CelebA 人脸生成项目

基于变分自编码器(VAE)的CelebA人脸数据集生成模型，采用模块化设计，易于使用和扩展。

## 🏗️ 项目结构

```
VAE/
├── config.py              # 超参数配置
├── dataset.py            # CustomCelebADataset数据集类
├── vae.py                # VAE模型架构(Encoder, Decoder, VAE)
├── train.py              # 训练逻辑和训练器
├── visualize.py          # 可视化和分析工具
├── main.py               # 项目主入口
├── README.md             # 说明文档
└── data/                 # 数据目录
    └── celeba/
        └── img_align_celeba/  # CelebA图像文件
```

## 🚀 快速开始

### 1. 环境要求

```bash
pip install torch torchvision matplotlib pillow scikit-learn
```

### 2. 数据准备

将CelebA数据集放置在 `data/celeba/img_align_celeba/` 目录下。

### 3. 运行方式

#### 方式一：交互模式（推荐）
```bash
python main.py
```

#### 方式二：命令行模式
```bash
# 训练模式
python main.py --mode train

# 可视化模式
python main.py --mode visualize

# 测试模式
python main.py --mode test

# 查看配置
python main.py --config
```

#### 方式三：直接运行模块
```bash
# 直接训练
python train.py

# 直接可视化
python visualize.py

# 测试模型架构
python vae.py

# 测试数据加载
python dataset.py
```

## ⚙️ 配置说明

在 `config.py` 中可以调整以下参数：

```python
class Config:
    # 设备配置
    device = "mps"  # 或 "cuda" / "cpu"
    
    # 数据配置
    batch_size = 32
    image_size = 64
    
    # 模型配置
    latent_dim = 128
    
    # 训练配置
    epochs = 50
    learning_rate = 1e-3
```

## 📊 功能特性

### 训练功能
- ✅ 模块化训练器
- ✅ 实时训练进度显示
- ✅ 自动模型保存
- ✅ 训练过程样本生成
- ✅ 损失曲线记录

### 可视化功能
- ✅ 潜在空间分布可视化
- ✅ 随机样本生成
- ✅ 图像重建对比
- ✅ 潜在空间插值
- ✅ 训练曲线绘制

### 模型架构
- ✅ 卷积编码器 (3×64×64 → 128维)
- ✅ 反卷积解码器 (128维 → 3×64×64)
- ✅ 重参数化技巧
- ✅ Beta-VAE支持

## 🎯 使用示例

### 快速训练
```python
from train import quick_train
from config import Config

# 调整配置
Config.epochs = 20
Config.batch_size = 64

# 开始训练
model, trainer = quick_train()
```

### 生成样本
```python
from vae import create_vae_model
from visualize import VAEVisualizer

# 加载模型
model = create_vae_model()
# ... 加载权重 ...

# 创建可视化器
visualizer = VAEVisualizer(model)

# 生成16个样本
visualizer.generate_samples(num_samples=16)
```

### 潜在空间分析
```python
# 可视化潜在空间分布
visualizer.visualize_latent_space(method='pca')

# 潜在空间插值
visualizer.latent_space_interpolation(num_steps=10)
```

## 📈 训练建议

### 快速训练（1小时内）
```python
Config.batch_size = 64    # 增加批次大小
Config.epochs = 15        # 减少训练轮数
Config.latent_dim = 64    # 减少潜在维度
Config.learning_rate = 2e-3  # 提高学习率
```

### 高质量训练
```python
Config.batch_size = 32    # 标准批次大小
Config.epochs = 50        # 充分训练
Config.latent_dim = 128   # 更大潜在空间
Config.learning_rate = 1e-3  # 稳定学习率
```

## 🛠️ 自定义扩展

### 添加新的损失函数
```python
def custom_loss_fn(x_hat, x, mu, logvar):
    # 自定义损失计算
    pass
```

### 修改模型架构
```python
class CustomEncoder(nn.Module):
    # 自定义编码器架构
    pass
```

### 新增可视化功能
```python
class CustomVisualizer(VAEVisualizer):
    def custom_plot(self):
        # 自定义可视化
        pass
```

## 🔧 故障排除

### 常见问题

1. **内存不足**
   - 减小 `batch_size`
   - 减小 `latent_dim`

2. **训练速度慢**
   - 增加 `batch_size`
   - 减少 `epochs`
   - 使用GPU加速

3. **生成质量差**
   - 增加 `epochs`
   - 调整 `learning_rate`
   - 增加 `latent_dim`

4. **数据加载失败**
   - 检查数据路径：`data/celeba/img_align_celeba/`
   - 确认图像文件格式为JPG

## 📝 更新日志

- **v1.0** - 初始版本，基础VAE实现
- **v2.0** - 模块化重构，增加交互式界面
- **v2.1** - 优化训练流程，增加可视化功能

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

MIT License