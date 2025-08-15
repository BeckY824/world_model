"""
VMC数据集模块 - 使用MNIST进行演示
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from config import VMCConfig

class VMCDataset:
    """VMC数据集管理器"""
    
    def __init__(self):
        self.config = VMCConfig()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化
        ])
    
    def create_dataloaders(self):
        """创建训练和测试数据加载器"""
        # 训练数据集
        train_dataset = datasets.MNIST(
            root='./data', 
            train=True,
            download=True, 
            transform=self.transform
        )
        
        # 测试数据集
        test_dataset = datasets.MNIST(
            root='./data', 
            train=False,
            download=True, 
            transform=self.transform
        )
        
        # 数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # M1兼容性
            pin_memory=False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"✅ 数据集加载完成")
        print(f"   训练样本数: {len(train_dataset)}")
        print(f"   测试样本数: {len(test_dataset)}")
        print(f"   批次大小: {self.config.batch_size}")
        print(f"   训练批次数: {len(train_loader)}")
        print(f"   测试批次数: {len(test_loader)}")
        
        return train_loader, test_loader
    
    def visualize_samples(self, dataloader, save_path="results/data_samples.png"):
        """可视化数据样本"""
        import os
        
        # 获取一个批次的数据
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
        
        # 选择前16个样本进行可视化
        sample_images = images[:16]
        sample_labels = labels[:16]
        
        # 创建4x4的子图
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle('MNIST Dataset Samples', fontsize=16)
        
        for i, (ax, img, label) in enumerate(zip(axes.flat, sample_images, sample_labels)):
            # 将图像从tensor转换并去标准化
            img_np = img.squeeze().numpy()
            img_np = img_np * 0.3081 + 0.1307  # 反标准化
            img_np = np.clip(img_np, 0, 1)
            
            ax.imshow(img_np, cmap='gray')
            ax.set_title(f'Label: {label.item()}')
            ax.axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 数据样本可视化已保存: {save_path}")
    
    def get_sample_batch(self, dataloader):
        """获取一个样本批次用于测试"""
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
        
        # 将图像展平为向量
        images_flat = images.view(images.size(0), -1)  # [batch_size, 784]
        
        return images_flat.to(self.config.device), labels.to(self.config.device)

def test_dataset():
    """测试数据集模块"""
    print("🧪 测试VMC数据集模块...")
    
    # 创建数据集管理器
    dataset_manager = VMCDataset()
    
    # 创建数据加载器
    train_loader, test_loader = dataset_manager.create_dataloaders()
    
    # 可视化样本
    dataset_manager.visualize_samples(train_loader)
    
    # 测试获取样本批次
    sample_images, sample_labels = dataset_manager.get_sample_batch(train_loader)
    print(f"样本批次形状: 图像 {sample_images.shape}, 标签 {sample_labels.shape}")
    
    print("✅ 数据集模块测试通过！")

if __name__ == "__main__":
    test_dataset()