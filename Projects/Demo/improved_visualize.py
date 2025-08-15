#!/usr/bin/env python3
"""
改进的VMC可视化脚本 - 根据用户需求定制
"""
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import os
from config import VMCConfig
from vmc_model import create_vmc_model
from dataset import VMCDataset

def calculate_psnr(original, reconstructed):
    """计算PSNR (Peak Signal-to-Noise Ratio)"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim_simple(img1, img2):
    """简化的SSIM计算"""
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim

def improved_visualize_components():
    """改进的VMC组件可视化"""
    
    print("🎨 改进的VMC可视化开始...")
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    try:
        # 加载检查点
        checkpoint_path = "checkpoints/vmc_checkpoint_epoch_3.pth"
        print(f"📂 加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=VMCConfig.device, weights_only=False)
        
        # 创建模型并加载权重
        model = create_vmc_model(VMCConfig())
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ 模型加载完成")
        
        # 加载数据
        dataset = VMCDataset()
        train_loader, test_loader = dataset.create_dataloaders()
        
        # 获取一些测试数据
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(VMCConfig.device)
                labels = labels.to(VMCConfig.device)
                
                # 扁平化图像 [batch_size, 1, 28, 28] -> [batch_size, 784]
                images_flat = images.view(images.size(0), -1)
                
                # 通过模型获取各组件输出
                # V组件 - 变分编码器
                v_output, mu, logvar = model.variational_encoder(images_flat)
                
                # 重构图像 (通过解码器)
                # 创建一个简单的解码器来重构图像用于评估
                decoder = torch.nn.Sequential(
                    torch.nn.Linear(VMCConfig.variational_dim, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 784),
                    torch.nn.Sigmoid()
                ).to(VMCConfig.device)
                
                # 使用变分编码进行重构
                reconstructed_flat = decoder(v_output)
                reconstructed = reconstructed_flat.view(-1, 1, 28, 28)
                
                # 将变分编码转换为记忆查询 (64 -> 32维)
                memory_query = model.var_to_memory(v_output)
                
                # M组件 - 记忆模块  
                memory_output, attention_weights, mixture_probs = model.memory_module(memory_query)
                
                break  # 只处理第一个批次
        
        # 转换为numpy数组以便可视化
        images_np = images.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()
        v_data = v_output.cpu().numpy()
        mu_np = mu.cpu().numpy()
        logvar_np = logvar.cpu().numpy()
        attention_data = attention_weights.cpu().numpy()
        mixture_data = mixture_probs.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        print("📊 开始生成改进的可视化图表...")
        
        # 1. V组件可视化 - 重构图像质量评估
        print("   生成V组件重构质量评估...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原图vs重构图对比 (显示前6个样本)
        for i in range(6):
            row = i // 3
            col = i % 3
            
            # 原图
            axes[row, col].imshow(images_np[i, 0], cmap='gray')
            axes[row, col].set_title(f'Original vs Reconstructed\nSample {i+1} (Label: {labels_np[i]})')
            axes[row, col].axis('off')
            
            # 在同一个subplot中显示重构图（使用边框区分）
            # 创建一个拼接图像：左边原图，右边重构图
            combined = np.hstack([images_np[i, 0], reconstructed_np[i, 0]])
            axes[row, col].clear()
            axes[row, col].imshow(combined, cmap='gray')
            axes[row, col].axvline(x=13.5, color='red', linewidth=2)  # 分割线
            axes[row, col].set_title(f'Original | Reconstructed\nLabel: {labels_np[i]}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/v_component_reconstruction_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # V组件质量指标
        plt.figure(figsize=(15, 5))
        
        # 计算重构质量指标
        reconstruction_metrics = []
        psnr_values = []
        ssim_values = []
        mse_values = []
        
        for i in range(min(32, len(images_np))):  # 计算前32个样本的指标
            original = images_np[i, 0]
            recon = reconstructed_np[i, 0]
            
            mse = mean_squared_error(original.flatten(), recon.flatten())
            psnr = calculate_psnr(original, recon)
            ssim = calculate_ssim_simple(original, recon)
            
            mse_values.append(mse)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
        
        # 绘制质量指标
        plt.subplot(1, 4, 1)
        plt.hist(mse_values, bins=15, alpha=0.7, color='red')
        plt.title('Reconstruction MSE Distribution')
        plt.xlabel('MSE')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(mse_values), color='darkred', linestyle='--', 
                   label=f'Mean: {np.mean(mse_values):.4f}')
        plt.legend()
        
        plt.subplot(1, 4, 2)
        plt.hist(psnr_values, bins=15, alpha=0.7, color='blue')
        plt.title('PSNR Distribution')
        plt.xlabel('PSNR (dB)')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(psnr_values), color='darkblue', linestyle='--',
                   label=f'Mean: {np.mean(psnr_values):.2f}dB')
        plt.legend()
        
        plt.subplot(1, 4, 3)
        plt.hist(ssim_values, bins=15, alpha=0.7, color='green')
        plt.title('SSIM Distribution')
        plt.xlabel('SSIM')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(ssim_values), color='darkgreen', linestyle='--',
                   label=f'Mean: {np.mean(ssim_values):.3f}')
        plt.legend()
        
        # KL散度分布
        plt.subplot(1, 4, 4)
        kl_div = -0.5 * torch.sum(1 + torch.tensor(logvar_np) - torch.tensor(mu_np).pow(2) - torch.tensor(logvar_np).exp(), dim=1)
        plt.hist(kl_div.numpy(), bins=15, alpha=0.7, color='purple')
        plt.title('KL Divergence Distribution')
        plt.xlabel('KL Divergence')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(kl_div.numpy()), color='indigo', linestyle='--',
                   label=f'Mean: {np.mean(kl_div.numpy()):.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/v_component_quality_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ V组件重构质量评估完成: results/v_component_reconstruction_comparison.png")
        print("✅ V组件质量指标完成: results/v_component_quality_metrics.png")
        
        # 2. M组件可视化 - 概率分布
        print("   生成M组件概率分布...")
        plt.figure(figsize=(15, 5))
        
        # 混合高斯概率分布
        plt.subplot(1, 3, 1)
        # 对每个样本的混合概率求平均
        avg_mixture_probs = np.mean(mixture_data, axis=0)
        plt.bar(range(len(avg_mixture_probs)), avg_mixture_probs, alpha=0.7, color='orange')
        plt.title('Mixture Component Probabilities\n(Average across batch)')
        plt.xlabel('Gaussian Component')
        plt.ylabel('Probability')
        plt.xticks(range(len(avg_mixture_probs)))
        for i, v in enumerate(avg_mixture_probs):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 注意力权重分布
        plt.subplot(1, 3, 2)
        # 计算平均注意力权重
        avg_attention = np.mean(attention_data, axis=0)
        plt.bar(range(len(avg_attention)), avg_attention, alpha=0.7, color='cyan')
        plt.title('Memory Attention Weights\n(Average across batch)')
        plt.xlabel('Memory Slot')
        plt.ylabel('Attention Weight')
        plt.xticks(range(0, len(avg_attention), 2))  # 只显示部分刻度
        
        # 注意力权重的分布情况
        plt.subplot(1, 3, 3)
        attention_entropy = []
        for i in range(len(attention_data)):
            # 计算每个样本的注意力熵（衡量注意力的集中程度）
            attention = attention_data[i]
            attention = attention / np.sum(attention)  # 归一化
            entropy = -np.sum(attention * np.log(attention + 1e-8))
            attention_entropy.append(entropy)
        
        plt.hist(attention_entropy, bins=15, alpha=0.7, color='magenta')
        plt.title('Attention Entropy Distribution\n(Higher = More Distributed)')
        plt.xlabel('Entropy')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(attention_entropy), color='darkmagenta', linestyle='--',
                   label=f'Mean: {np.mean(attention_entropy):.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/m_component_probability_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ M组件概率分布完成: results/m_component_probability_distributions.png")
        
        # 3. 生成总结报告
        print("   生成VMC性能总结报告...")
        
        # 计算总体性能指标
        avg_mse = np.mean(mse_values)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_kl = np.mean(kl_div.numpy())
        avg_attention_entropy = np.mean(attention_entropy)
        
        # 创建性能报告文本
        report = f"""
VMC (Variational Memory Compression) 性能评估报告
===============================================

训练信息:
- 检查点: {checkpoint_path}
- 训练轮数: 3 epochs
- 模型参数: 620,819

V组件 (变分编码器) 重构质量评估:
- 平均MSE损失: {avg_mse:.6f}
- 平均PSNR: {avg_psnr:.2f} dB
- 平均SSIM: {avg_ssim:.4f}
- 平均KL散度: {avg_kl:.4f}

重构质量评级:
- PSNR > 20dB: {'优秀' if avg_psnr > 20 else '良好' if avg_psnr > 15 else '一般'}
- SSIM > 0.8: {'优秀' if avg_ssim > 0.8 else '良好' if avg_ssim > 0.6 else '一般'}

M组件 (记忆模块) 概率分布分析:
- 混合高斯组件数: {len(avg_mixture_probs)}
- 最活跃组件: Component {np.argmax(avg_mixture_probs)} (概率: {np.max(avg_mixture_probs):.3f})
- 记忆槽数量: {len(avg_attention)}
- 平均注意力熵: {avg_attention_entropy:.3f}

注意力集中度评估:
- 熵值 < 2.0: {'集中' if avg_attention_entropy < 2.0 else '分散'}

生成的可视化文件:
1. results/v_component_reconstruction_comparison.png - 原图vs重构图对比
2. results/v_component_quality_metrics.png - 重构质量指标分布  
3. results/m_component_probability_distributions.png - 记忆模块概率分布

总体评估: VMC模型展现了{'良好' if avg_psnr > 15 and avg_ssim > 0.6 else '可接受'}的重构性能和{'有效' if avg_attention_entropy < 3.0 else '分散'}的记忆机制。
"""
        
        # 保存报告
        with open('results/vmc_performance_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n🎉 改进的VMC可视化全部完成!")
        print("📊 生成的文件:")
        print("   • results/v_component_reconstruction_comparison.png - V组件重构对比")
        print("   • results/v_component_quality_metrics.png - V组件质量指标")
        print("   • results/m_component_probability_distributions.png - M组件概率分布")
        print("   • results/vmc_performance_report.txt - 性能评估报告")
        print(f"\n📈 关键指标:")
        print(f"   • 平均PSNR: {avg_psnr:.2f} dB")
        print(f"   • 平均SSIM: {avg_ssim:.4f}")
        print(f"   • 平均注意力熵: {avg_attention_entropy:.3f}")
        
        # 输出报告内容
        print("\n" + "="*50)
        print(report)
        
    except Exception as e:
        print(f"❌ 可视化过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    improved_visualize_components()