"""
训练模块 - VAE训练逻辑
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
from datetime import datetime

from config import Config
from vae import create_vae_model, vae_loss_fn
from dataset import create_dataloader

class VAETrainer:
    """VAE训练器"""
    
    def __init__(self, model=None, dataloader=None):
        """
        初始化训练器
        
        Args:
            model: VAE模型，如果为None则自动创建
            dataloader: 数据加载器，如果为None则自动创建
        """
        self.device = Config.device
        self.model = model if model is not None else create_vae_model()
        self.dataloader = dataloader if dataloader is not None else create_dataloader()
        
        if self.dataloader is None:
            raise ValueError("无法创建数据加载器")
            
        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.learning_rate)
        
        # 训练统计
        self.train_losses = []
        self.recon_losses = []
        self.kl_losses = []
        
        print(f"🚀 VAE训练器初始化完成")
        print(f"   模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   数据集大小: {len(self.dataloader.dataset)}")
        print(f"   批次数量: {len(self.dataloader)}")
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch编号
            
        Returns:
            epoch_loss: 平均损失
            epoch_recon_loss: 平均重建损失
            epoch_kl_loss: 平均KL损失
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch_idx, (x, _) in enumerate(self.dataloader):
            x = x.to(self.device)
            
            # 前向传播
            x_hat, mu, logvar = self.model(x)
            
            # 计算损失
            loss, recon_loss, kl_loss = vae_loss_fn(x_hat, x, mu, logvar)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # 打印进度
            if batch_idx % Config.print_interval == 0:
                print(f'  Epoch {epoch+1}/{Config.epochs}, '
                      f'Batch {batch_idx}/{len(self.dataloader)}, '
                      f'Loss: {loss.item():.2f}')
        
        # 计算平均损失
        epoch_loss = total_loss / len(self.dataloader.dataset)
        epoch_recon_loss = total_recon_loss / len(self.dataloader.dataset)
        epoch_kl_loss = total_kl_loss / len(self.dataloader.dataset)
        
        return epoch_loss, epoch_recon_loss, epoch_kl_loss
    
    def save_samples(self, epoch, save_dir="results"):
        """
        保存生成样本
        
        Args:
            epoch: 当前epoch
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.eval()
        with torch.no_grad():
            # 生成样本
            samples = self.model.sample(Config.num_samples, self.device)
            
            # 反归一化到[0,1]
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            
            # 创建网格
            grid = make_grid(samples, nrow=Config.grid_size, pad_value=1)
            
            # 保存图像
            plt.figure(figsize=(8, 8))
            plt.imshow(grid.permute(1, 2, 0).cpu())
            plt.axis('off')
            plt.title(f'Generated Faces - Epoch {epoch+1}')
            
            filename = os.path.join(save_dir, f'generated_faces_epoch_{epoch+1}.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  💾 样本已保存: {filename}")
        
        self.model.train()
    
    def train(self):
        """开始训练"""
        print(f"🎯 开始训练VAE模型")
        print(f"   训练轮数: {Config.epochs}")
        print(f"   学习率: {Config.learning_rate}")
        print(f"   批次大小: {Config.batch_size}")
        print("=" * 60)
        
        start_time = datetime.now()
        
        for epoch in range(Config.epochs):
            print(f"📈 Epoch {epoch+1}/{Config.epochs}")
            
            # 训练一个epoch
            epoch_loss, recon_loss, kl_loss = self.train_epoch(epoch)
            
            # 记录损失
            self.train_losses.append(epoch_loss)
            self.recon_losses.append(recon_loss)
            self.kl_losses.append(kl_loss)
            
            # 打印统计
            print(f"  ✅ Epoch {epoch+1} 完成")
            print(f"     总损失: {epoch_loss:.4f}")
            print(f"     重建损失: {recon_loss:.4f}")
            print(f"     KL损失: {kl_loss:.4f}")
            
            # 保存样本
            if (epoch + 1) % Config.save_interval == 0:
                self.save_samples(epoch)
            
            print("-" * 40)
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print("🎉 训练完成！")
        print(f"   总训练时间: {training_time}")
        print(f"   最终损失: {self.train_losses[-1]:.4f}")
        
        return self.model
    
    def save_model(self, filepath="vae_model.pth"):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'recon_losses': self.recon_losses,
            'kl_losses': self.kl_losses,
            'config': {
                'latent_dim': Config.latent_dim,
                'learning_rate': Config.learning_rate,
                'batch_size': Config.batch_size,
                'epochs': Config.epochs
            }
        }, filepath)
        print(f"💾 模型已保存: {filepath}")
    
    def load_model(self, filepath="vae_model.pth"):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.recon_losses = checkpoint['recon_losses']
            self.kl_losses = checkpoint['kl_losses']
            
        print(f"📂 模型已加载: {filepath}")

def quick_train():
    """快速训练函数"""
    try:
        # 创建训练器
        trainer = VAETrainer()
        
        # 开始训练
        trained_model = trainer.train()
        
        # 保存模型
        trainer.save_model()
        
        return trained_model, trainer
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        return None, None

if __name__ == "__main__":
    # 直接运行训练
    print("启动VAE训练...")
    Config.print_config()
    
    trained_model, trainer = quick_train()
    
    if trained_model is not None:
        print("✅ 训练成功完成！")
    else:
        print("❌ 训练失败！")