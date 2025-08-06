"""
VMC (Variational Memory Compression) 配置文件
针对MacBook Air M1 16GB内存优化
"""
import torch

class VMCConfig:
    """VMC模型配置"""
    
    # 设备配置 - 针对M1芯片优化
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 数据配置
    batch_size = 32  # 适合16GB内存
    seq_length = 50  # 序列长度
    input_dim = 784  # 输入维度 (例如28x28 MNIST)
    
    # V - Variational Encoder 配置
    variational_dim = 64  # 变分潜在空间维度
    encoder_hidden_dims = [512, 256, 128]  # 编码器隐藏层维度
    
    # M - Memory 配置
    memory_size = 16  # 记忆槽数量
    memory_dim = 32   # 每个记忆槽的维度
    num_gaussians = 8  # 混合高斯分布的组件数
    
    # C - Controller 配置
    controller_hidden_dim = 128
    controller_output_dim = 10  # 分类任务输出维度
    
    # 训练配置
    epochs = 30  # 适中的训练轮数
    learning_rate = 1e-3
    kl_beta = 0.1  # KL散度权重
    memory_beta = 0.01  # 记忆损失权重
    
    # 可视化配置
    save_interval = 5  # 每5个epoch保存可视化结果
    vis_samples = 8    # 可视化样本数量
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 60)
        print("VMC (Variational Memory Compression) 配置")
        print("=" * 60)
        print(f"设备: {cls.device}")
        print(f"批次大小: {cls.batch_size}")
        print(f"序列长度: {cls.seq_length}")
        print(f"输入维度: {cls.input_dim}")
        print(f"变分维度: {cls.variational_dim}")
        print(f"记忆槽数量: {cls.memory_size}")
        print(f"记忆维度: {cls.memory_dim}")
        print(f"混合高斯组件数: {cls.num_gaussians}")
        print(f"训练轮数: {cls.epochs}")
        print(f"学习率: {cls.learning_rate}")
        print("=" * 60)