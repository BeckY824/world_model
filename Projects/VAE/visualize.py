"""
可视化模块 - VAE结果可视化和分析
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from config import Config
from vae import create_vae_model
from dataset import create_dataloader

class VAEVisualizer:
    """VAE可视化器"""
    
    def __init__(self, model=None, dataloader=None):
        """
        初始化可视化器
        
        Args:
            model: 训练好的VAE模型
            dataloader: 数据加载器
        """
        self.device = Config.device
        self.model = model if model is not None else create_vae_model()
        self.dataloader = dataloader if dataloader is not None else create_dataloader()
        
        if self.dataloader is None:
            raise ValueError("无法创建数据加载器")
            
        self.model.eval()
        print("🎨 VAE可视化器初始化完成")
    
    def visualize_latent_space(self, num_samples=1000, method='pca', save_path="results/latent_space.png"):
        """
        可视化潜在空间分布
        
        Args:
            num_samples: 采样数量
            method: 降维方法 ('pca' 或 'tsne')
            save_path: 保存路径
        """
        print(f"📊 可视化潜在空间分布 (方法: {method})")
        
        # 收集潜在空间编码
        mu_list = []
        
        with torch.no_grad():
            sample_count = 0
            for x, _ in self.dataloader:
                if sample_count >= num_samples:
                    break
                    
                x = x.to(self.device)
                mu, _ = self.model.encoder(x)
                mu_list.append(mu.cpu())
                sample_count += x.size(0)
        
        # 合并所有编码
        mu_all = torch.cat(mu_list)[:num_samples].numpy()
        print(f"   收集了 {mu_all.shape[0]} 个样本的潜在编码")
        
        # 降维到2D
        if method == 'pca':
            reducer = PCA(n_components=2)
            mu_2d = reducer.fit_transform(mu_all)
            title = f"Latent Space (PCA) - {mu_all.shape[1]}D → 2D"
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            mu_2d = reducer.fit_transform(mu_all)
            title = f"Latent Space (t-SNE) - {mu_all.shape[1]}D → 2D"
        else:
            # 如果潜在空间维度>=2，直接使用前两个维度
            mu_2d = mu_all[:, :2]
            title = "Latent Space (First 2 Dimensions)"
        
        # 创建散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(mu_2d[:, 0], mu_2d[:, 1], alpha=0.6, s=1, c='blue')
        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"   💾 潜在空间可视化已保存: {save_path}")
    
    def generate_samples(self, num_samples=16, grid_size=4, save_path="results/generated_samples.png"):
        """
        生成并可视化样本
        
        Args:
            num_samples: 生成样本数量
            grid_size: 网格大小
            save_path: 保存路径
        """
        print(f"🎭 生成 {num_samples} 个样本")
        
        with torch.no_grad():
            # 从标准正态分布采样
            z = torch.randn(num_samples, self.model.latent_dim).to(self.device)
            samples = self.model.decoder(z)
            
            # 反归一化到[0,1]
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            
            # 创建网格
            grid = make_grid(samples, nrow=grid_size, pad_value=1)
            
            # 显示和保存
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0).cpu())
            plt.axis('off')
            plt.title('Generated Faces from Random Latent Codes')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"   💾 生成样本已保存: {save_path}")
    
    def show_reconstruction(self, num_pairs=8, save_path="results/reconstruction.png"):
        """
        显示重建对比
        
        Args:
            num_pairs: 对比对数
            save_path: 保存路径
        """
        print(f"🔄 显示重建对比 ({num_pairs} 对)")
        
        with torch.no_grad():
            # 获取真实图像
            x, _ = next(iter(self.dataloader))
            x = x[:num_pairs].to(self.device)
            
            # 重建图像
            x_hat, _, _ = self.model(x)
            
            # 反归一化
            x_display = (x.cpu() + 1) / 2
            x_hat_display = (x_hat.cpu() + 1) / 2
            x_display = torch.clamp(x_display, 0, 1)
            x_hat_display = torch.clamp(x_hat_display, 0, 1)
            
            # 交替排列原图和重建图
            comparison = torch.zeros(2 * num_pairs, 3, Config.image_size, Config.image_size)
            for i in range(num_pairs):
                comparison[2*i] = x_display[i]      # 原图
                comparison[2*i+1] = x_hat_display[i]  # 重建图
            
            # 创建网格
            grid = make_grid(comparison, nrow=2, pad_value=1)
            
            # 显示和保存
            plt.figure(figsize=(15, 2 * num_pairs))
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.title('Reconstruction Comparison (Original | Reconstructed)')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"   💾 重建对比已保存: {save_path}")
    
    def latent_space_interpolation(self, num_steps=10, save_path="results/interpolation.png"):
        """
        潜在空间插值可视化
        
        Args:
            num_steps: 插值步数
            save_path: 保存路径
        """
        print(f"🌈 潜在空间插值 ({num_steps} 步)")
        
        with torch.no_grad():
            # 随机选择两个潜在点
            z1 = torch.randn(1, self.model.latent_dim).to(self.device)
            z2 = torch.randn(1, self.model.latent_dim).to(self.device)
            
            # 线性插值
            alphas = torch.linspace(0, 1, num_steps).to(self.device)
            interpolated_samples = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                sample = self.model.decoder(z_interp)
                interpolated_samples.append(sample)
            
            # 合并所有插值样本
            samples = torch.cat(interpolated_samples, dim=0)
            
            # 反归一化
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            
            # 创建网格
            grid = make_grid(samples, nrow=num_steps, pad_value=1)
            
            # 显示和保存
            plt.figure(figsize=(2 * num_steps, 4))
            plt.imshow(grid.permute(1, 2, 0).cpu())
            plt.axis('off')
            plt.title('Latent Space Interpolation')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"   💾 插值结果已保存: {save_path}")
    
    def plot_training_curves(self, train_losses, recon_losses, kl_losses, save_path="results/training_curves.png"):
        """
        绘制训练曲线
        
        Args:
            train_losses: 总训练损失
            recon_losses: 重建损失
            kl_losses: KL损失
            save_path: 保存路径
        """
        print("📈 绘制训练曲线")
        
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(15, 5))
        
        # 总损失
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses, 'b-', label='Total Loss')
        plt.title('Total Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 重建损失
        plt.subplot(1, 3, 2)
        plt.plot(epochs, recon_losses, 'r-', label='Reconstruction Loss')
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # KL损失
        plt.subplot(1, 3, 3)
        plt.plot(epochs, kl_losses, 'g-', label='KL Loss')
        plt.title('KL Divergence Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"   💾 训练曲线已保存: {save_path}")
    
    def comprehensive_analysis(self, results_dir="results"):
        """
        综合分析和可视化
        
        Args:
            results_dir: 结果保存目录
        """
        print("🔍 开始综合分析...")
        
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. 潜在空间可视化
        self.visualize_latent_space(
            save_path=os.path.join(results_dir, "latent_space_pca.png"),
            method='pca'
        )
        
        # 2. 生成样本
        self.generate_samples(
            save_path=os.path.join(results_dir, "generated_samples.png")
        )
        
        # 3. 重建对比
        self.show_reconstruction(
            save_path=os.path.join(results_dir, "reconstruction_comparison.png")
        )
        
        # 4. 潜在空间插值
        self.latent_space_interpolation(
            save_path=os.path.join(results_dir, "latent_interpolation.png")
        )
        
        print(f"✅ 综合分析完成，结果保存在: {results_dir}")

def load_and_visualize(model_path="vae_model.pth"):
    """
    加载模型并进行可视化
    
    Args:
        model_path: 模型文件路径
    """
    # 创建模型
    model = create_vae_model()
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=Config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"📂 已加载模型: {model_path}")
    
    # 创建可视化器
    visualizer = VAEVisualizer(model)
    
    # 综合分析
    visualizer.comprehensive_analysis()
    
    # 如果有训练历史，绘制训练曲线
    if 'train_losses' in checkpoint:
        visualizer.plot_training_curves(
            checkpoint['train_losses'],
            checkpoint['recon_losses'], 
            checkpoint['kl_losses']
        )

if __name__ == "__main__":
    print("启动VAE可视化...")
    
    # 检查是否有训练好的模型
    if os.path.exists("vae_model.pth"):
        load_and_visualize("vae_model.pth")
    else:
        print("未找到训练好的模型，请先运行训练")
        print("您可以运行: python train.py")