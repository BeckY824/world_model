"""
配置文件 - VAE训练超参数
"""
import torch

class Config:
    """VAE训练配置"""
    
    # 设备配置
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 数据配置
    batch_size = 32
    image_size = 64
    data_root = "./data/celeba/img_align_celeba"
    
    # 模型配置
    latent_dim = 128
    
    # 训练配置
    epochs = 50
    learning_rate = 1e-3
    
    # 日志配置
    print_interval = 100  # 每多少个batch打印一次
    save_interval = 10    # 每多少个epoch保存一次样本
    
    # 可视化配置
    num_samples = 16      # 生成样本数量
    grid_size = 4         # 生成样本网格大小
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 50)
        print("VAE 训练配置")
        print("=" * 50)
        print(f"设备: {cls.device}")
        print(f"批次大小: {cls.batch_size}")
        print(f"图像尺寸: {cls.image_size}x{cls.image_size}")
        print(f"潜在空间维度: {cls.latent_dim}")
        print(f"训练轮数: {cls.epochs}")
        print(f"学习率: {cls.learning_rate}")
        print("=" * 50)