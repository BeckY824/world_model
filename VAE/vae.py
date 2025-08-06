"""
VAE模型架构 - 编码器、解码器和VAE类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class Encoder(nn.Module):
    """VAE编码器 - 将图像编码为潜在空间的均值和方差"""
    
    def __init__(self, latent_dim=None):
        super().__init__()
        if latent_dim is None:
            latent_dim = Config.latent_dim
            
        # 卷积层：3x64x64 -> 256x4x4
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)      # 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)     # 64x16x16
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)    # 128x8x8
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)   # 256x4x4
        
        # 全连接层：输出潜在空间的均值和方差
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 (batch_size, 3, 64, 64)
            
        Returns:
            mu: 潜在空间均值 (batch_size, latent_dim)
            logvar: 潜在空间对数方差 (batch_size, latent_dim)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.view(x.size(0), -1)  # 展平
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    """VAE解码器 - 将潜在空间编码重建为图像"""
    
    def __init__(self, latent_dim=None):
        super().__init__()
        if latent_dim is None:
            latent_dim = Config.latent_dim
            
        # 全连接层：潜在空间 -> 特征图
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # 反卷积层：256x4x4 -> 3x64x64
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 128x8x8
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 64x16x16
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 32x32x32
        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)     # 3x64x64

    def forward(self, z):
        """
        前向传播
        
        Args:
            z: 潜在空间编码 (batch_size, latent_dim)
            
        Returns:
            x_hat: 重建图像 (batch_size, 3, 64, 64)
        """
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), 256, 4, 4)  # 重塑为4D张量
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.tanh(self.deconv4(x))  # 使用tanh配合[-1,1]归一化
        return x

class VAE(nn.Module):
    """变分自编码器(VAE)主模型"""
    
    def __init__(self, latent_dim=None):
        super().__init__()
        if latent_dim is None:
            latent_dim = Config.latent_dim
            
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：从N(mu, var)采样
        
        Args:
            mu: 均值
            logvar: 对数方差
            
        Returns:
            z: 重参数化后的潜在变量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # 标准正态分布噪声
        return mu + eps * std

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 (batch_size, 3, 64, 64)
            
        Returns:
            x_hat: 重建图像 (batch_size, 3, 64, 64)
            mu: 潜在空间均值 (batch_size, latent_dim) 
            logvar: 潜在空间对数方差 (batch_size, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
    
    def sample(self, num_samples, device=None):
        """
        从潜在空间采样生成新图像
        
        Args:
            num_samples: 采样数量
            device: 设备
            
        Returns:
            生成的图像
        """
        if device is None:
            device = Config.device
            
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
            return samples

def vae_loss_fn(x_hat, x, mu, logvar, beta=1.0):
    """
    VAE损失函数
    
    Args:
        x_hat: 重建图像
        x: 原始图像
        mu: 潜在空间均值
        logvar: 潜在空间对数方差
        beta: KL损失权重 (Beta-VAE)
        
    Returns:
        total_loss: 总损失
        recon_loss: 重建损失
        kl_loss: KL散度损失
    """
    # 重建损失 - 使用MSE
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

def create_vae_model(latent_dim=None):
    """
    创建VAE模型
    
    Args:
        latent_dim: 潜在空间维度
        
    Returns:
        VAE模型实例
    """
    if latent_dim is None:
        latent_dim = Config.latent_dim
    
    model = VAE(latent_dim).to(Config.device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🏗️  VAE模型创建完成")
    print(f"   潜在空间维度: {latent_dim}")
    print(f"   总参数数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    return model

if __name__ == "__main__":
    # 测试模型创建
    print("测试VAE模型...")
    vae = create_vae_model()
    
    # 测试前向传播
    test_input = torch.randn(2, 3, 64, 64).to(Config.device)
    x_hat, mu, logvar = vae(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"重建输出形状: {x_hat.shape}")
    print(f"潜在均值形状: {mu.shape}")
    print(f"潜在方差形状: {logvar.shape}")
    print("✅ VAE模型测试通过！")

