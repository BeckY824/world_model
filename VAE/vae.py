import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os

# MPS支持
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# 超参数 - 针对MacBook Air M4 16GB优化
batch_size = 32  # 减小batch_size以适应内存限制
latent_dim = 128  # 增加潜在空间维度以更好地表示人脸
epochs = 50
learning_rate = 1e-3

# 数据预处理 - 适配CelebA
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # CelebA标准尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
])

# 数据集和数据加载器的创建
def create_dataloader():
    """创建CelebA数据加载器，使用手动下载的数据集"""
    try:
        # 使用手动下载的数据集，不自动下载
        dataset = datasets.CelebA(root='./data', split='train', transform=transform, download=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        print(f"成功加载CelebA数据集，共有 {len(dataset)} 张训练图像")
        return dataloader
    except FileNotFoundError as e:
        print(f"未找到CelebA数据集: {e}")
        print("请确保已将CelebA数据集解压到 ./data 目录下")
        print("期望的目录结构:")
        print("  ./data/celeba/")
        print("    ├── img_align_celeba/")
        print("    ├── list_attr_celeba.txt")
        print("    ├── list_bbox_celeba.txt")
        print("    ├── list_landmarks_align_celeba.txt")
        print("    └── list_eval_partition.txt")
        return None
    except Exception as e:
        print(f"加载CelebA数据集时出错: {e}")
        return None

# 编码器 - 使用卷积层适配CelebA 64x64x3图像
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # 输入: 3x64x64
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)      # 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)     # 64x16x16
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)    # 128x8x8
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)   # 256x4x4
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.view(x.size(0), -1)  # 展平
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# 解码器 - 使用反卷积层重建64x64x3图像
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # 反卷积层
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 128x8x8
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 64x16x16
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 32x32x32
        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)     # 3x64x64

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), 256, 4, 4)  # 重塑为4D张量
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.tanh(self.deconv4(x))  # 使用tanh配合[-1,1]归一化
        return x

# VAE整体结构
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # 随机噪声
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

# 损失函数 - 适配CelebA图像和tanh输出
def loss_fn(x_hat, x, mu, logvar, beta=1.0):
    # 重建损失 - 使用MSE因为我们使用tanh输出和[-1,1]归一化
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Beta-VAE: 可以调节KL损失的权重
    return recon_loss + beta * kl_loss

# 模型训练
vae = VAE(latent_dim).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# 创建数据加载器
print("创建数据加载器...")
dataloader = create_dataloader()
if dataloader is None:
    print("无法创建数据加载器，训练终止。")
    print("您可以稍后重新运行此脚本，或手动下载CelebA数据集。")
    exit(1)

print("开始训练...")
for epoch in range(epochs):
    vae.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, (x, _) in enumerate(dataloader):
        x = x.to(device)
        x_hat, mu, logvar = vae(x)
        
        # 计算损失
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        # 每100个batch打印一次进度
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, '
                  f'Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon_loss = total_recon_loss / len(dataloader.dataset)
    avg_kl_loss = total_kl_loss / len(dataloader.dataset)
    
    print(f"Epoch [{epoch+1}/{epochs}] - Total Loss: {avg_loss:.4f}, "
          f"Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
    
    # 每10个epoch保存一次生成的样本
    if (epoch + 1) % 10 == 0:
        vae.eval()
        with torch.no_grad():
            # 生成一些样本查看效果
            z = torch.randn(16, latent_dim).to(device)
            samples = vae.decoder(z).cpu()
            # 反归一化到[0,1]
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            
            grid = make_grid(samples, nrow=4, pad_value=1)
            plt.figure(figsize=(8, 8))
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.title(f'Generated Faces - Epoch {epoch+1}')
            plt.savefig(f'generated_faces_epoch_{epoch+1}.png')
            plt.close()
        vae.train()

print("训练完成！")


# 可视化和评估函数
def visualize_latent_space(vae, dataloader, num_samples=1000):
    """可视化潜在空间分布"""
    vae.eval()
    mu_list = []
    
    with torch.no_grad():
        sample_count = 0
        for x, _ in dataloader:
            if sample_count >= num_samples:
                break
            x = x.to(device)
            mu, _ = vae.encoder(x)
            mu_list.append(mu.cpu())
            sample_count += x.size(0)
    
    mu_all = torch.cat(mu_list)[:num_samples].numpy()
    
    # 如果潜在空间维度较高，只可视化前两个维度
    if latent_dim >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(mu_all[:, 0], mu_all[:, 1], alpha=0.6, s=1)
        plt.title("Latent Space Distribution (First 2 Dimensions)")
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.grid(True)
        plt.savefig("latent_space_distribution.png", dpi=150)
        plt.show()

def show_generated_faces(decoder, grid_size=8):
    """生成人脸样本"""
    decoder.eval()
    z = torch.randn(grid_size**2, latent_dim).to(device)
    with torch.no_grad():
        samples = decoder(z).cpu()
        # 反归一化到[0,1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
    
    grid = make_grid(samples, nrow=grid_size, pad_value=1)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Generated Faces from Random Latents")
    plt.savefig("generated_faces.png", dpi=150, bbox_inches='tight')
    plt.show()

def show_reconstruction(vae, dataloader, num_pairs=8):
    """显示原始图像和重建图像的对比"""
    vae.eval()
    with torch.no_grad():
        x, _ = next(iter(dataloader))
        x = x[:num_pairs].to(device)
        x_hat, _, _ = vae(x)
        
        # 反归一化
        x_display = (x.cpu() + 1) / 2
        x_hat_display = (x_hat.cpu() + 1) / 2
        x_display = torch.clamp(x_display, 0, 1)
        x_hat_display = torch.clamp(x_hat_display, 0, 1)
        
        # 拼接原始和重建图像
        comparison = torch.cat([x_display, x_hat_display], dim=0)
        grid = make_grid(comparison, nrow=num_pairs, pad_value=1)
        
        plt.figure(figsize=(15, 4))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title('Top: Original Images, Bottom: Reconstructed Images')
        plt.savefig("reconstruction_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()

# 训练完成后进行可视化
if dataloader is not None:
    print("生成可视化结果...")
    visualize_latent_space(vae, dataloader)
    show_generated_faces(vae.decoder)
    show_reconstruction(vae, dataloader)
    print("所有可视化完成！")
else:
    print("由于数据加载器不可用，跳过可视化。")