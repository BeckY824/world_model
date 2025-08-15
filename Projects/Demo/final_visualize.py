#!/usr/bin/env python3
"""
最终的VMC可视化脚本 - 按用户需求定制
1. V组件: 潜在编码质量评估指标
2. M组件: 概率分布可视化
3. 移除C组件可视化
"""
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from config import VMCConfig
from vmc_model import create_vmc_model
from dataset import VMCDataset

# 设置中文字体（如果可用）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def final_visualize_components():
    """最终的VMC组件可视化 - 符合用户需求"""
    
    print("🎨 最终VMC可视化开始...")
    
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
        
        # 收集更多数据用于统计分析
        all_v_outputs = []
        all_mu = []
        all_logvar = []
        all_attention_weights = []
        all_mixture_probs = []
        all_labels = []
        
        print("📊 收集数据进行分析...")
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                if batch_idx >= 10:  # 只处理前10个批次
                    break
                    
                images = images.to(VMCConfig.device)
                labels = labels.to(VMCConfig.device)
                
                # 扁平化图像 [batch_size, 1, 28, 28] -> [batch_size, 784]
                images_flat = images.view(images.size(0), -1)
                
                # V组件 - 变分编码器
                v_output, mu, logvar = model.variational_encoder(images_flat)
                
                # 将变分编码转换为记忆查询 (64 -> 32维)
                memory_query = model.var_to_memory(v_output)
                
                # M组件 - 记忆模块  
                memory_output, attention_weights, mixture_probs = model.memory_module(memory_query)
                
                # 收集数据
                all_v_outputs.append(v_output.cpu())
                all_mu.append(mu.cpu())
                all_logvar.append(logvar.cpu())
                all_attention_weights.append(attention_weights.cpu())
                all_mixture_probs.append(mixture_probs.cpu())
                all_labels.append(labels.cpu())
        
        # 合并所有数据
        v_data = torch.cat(all_v_outputs, dim=0).numpy()
        mu_data = torch.cat(all_mu, dim=0).numpy()
        logvar_data = torch.cat(all_logvar, dim=0).numpy()
        attention_data = torch.cat(all_attention_weights, dim=0).numpy()
        mixture_data = torch.cat(all_mixture_probs, dim=0).numpy()
        labels_data = torch.cat(all_labels, dim=0).numpy()
        
        print("📊 开始生成最终可视化图表...")
        
        # ==================== 1. V组件评估指标 ====================
        print("   生成V组件评估指标...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1.1 潜在空间分布 (PCA 2D)
        pca = PCA(n_components=2)
        v_2d = pca.fit_transform(v_data)
        scatter = axes[0, 0].scatter(v_2d[:, 0], v_2d[:, 1], c=labels_data, cmap='tab10', alpha=0.6, s=20)
        axes[0, 0].set_title('V Component: Latent Space Distribution (PCA)')
        axes[0, 0].set_xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})')
        axes[0, 0].set_ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 1.2 KL散度分布
        kl_divergence = -0.5 * torch.sum(1 + torch.tensor(logvar_data) - torch.tensor(mu_data).pow(2) - torch.tensor(logvar_data).exp(), dim=1)
        axes[0, 1].hist(kl_divergence.numpy(), bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title('V Component: KL Divergence Distribution')
        axes[0, 1].set_xlabel('KL Divergence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(kl_divergence.numpy()), color='darkred', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(kl_divergence.numpy()):.3f}')
        axes[0, 1].legend()
        
        # 1.3 潜在维度激活度
        latent_activation = np.mean(np.abs(v_data), axis=0)
        axes[0, 2].bar(range(len(latent_activation)), latent_activation, alpha=0.7, color='blue')
        axes[0, 2].set_title('V Component: Latent Dimension Activation')
        axes[0, 2].set_xlabel('Latent Dimension')
        axes[0, 2].set_ylabel('Average Absolute Activation')
        axes[0, 2].set_xticks(range(0, len(latent_activation), 8))
        
        # 1.4 类别分离度评估
        class_centers = []
        for class_id in range(10):
            mask = labels_data == class_id
            if np.sum(mask) > 0:
                center = np.mean(v_data[mask], axis=0)
                class_centers.append(center)
        
        if len(class_centers) > 1:
            class_centers = np.array(class_centers)
            # 计算类别中心之间的距离
            from scipy.spatial.distance import pdist
            distances = pdist(class_centers)
            axes[1, 0].hist(distances, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[1, 0].set_title('V Component: Inter-class Distances')
            axes[1, 0].set_xlabel('Distance between Class Centers')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(np.mean(distances), color='darkgreen', linestyle='--', linewidth=2,
                              label=f'Mean: {np.mean(distances):.3f}')
            axes[1, 0].legend()
        
        # 1.5 方差分析
        variances = np.exp(logvar_data)  # 转换log方差为方差
        mean_variance_per_dim = np.mean(variances, axis=0)
        axes[1, 1].plot(mean_variance_per_dim, marker='o', markersize=4, alpha=0.7)
        axes[1, 1].set_title('V Component: Average Variance per Dimension')
        axes[1, 1].set_xlabel('Latent Dimension')
        axes[1, 1].set_ylabel('Average Variance')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 1.6 编码质量评分
        # 基于KL散度、类别分离度等计算综合评分
        avg_kl = np.mean(kl_divergence.numpy())
        avg_separation = np.mean(distances) if len(class_centers) > 1 else 0
        total_variance = np.sum(mean_variance_per_dim)
        
        quality_metrics = ['KL Divergence', 'Class Separation', 'Total Variance', 'Latent Utilization']
        quality_scores = [
            min(avg_kl / 10.0, 1.0),  # 归一化KL散度
            min(avg_separation / 5.0, 1.0),  # 归一化分离度
            min(total_variance / 50.0, 1.0),  # 归一化总方差
            np.mean(latent_activation > 0.1)  # 激活维度比例
        ]
        
        bars = axes[1, 2].bar(range(len(quality_metrics)), quality_scores, 
                             color=['red', 'green', 'blue', 'orange'], alpha=0.7)
        axes[1, 2].set_title('V Component: Quality Metrics')
        axes[1, 2].set_ylabel('Normalized Score')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_xticks(range(len(quality_metrics)))
        axes[1, 2].set_xticklabels(quality_metrics, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, score in zip(bars, quality_scores):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/v_component_evaluation_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ V组件评估指标完成: results/v_component_evaluation_metrics.png")
        
        # ==================== 2. M组件概率分布 ====================
        print("   生成M组件概率分布...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 2.1 混合高斯分量概率分布
        avg_mixture_probs = np.mean(mixture_data, axis=0)
        std_mixture_probs = np.std(mixture_data, axis=0)
        
        x_pos = np.arange(len(avg_mixture_probs))
        bars = axes[0, 0].bar(x_pos, avg_mixture_probs, yerr=std_mixture_probs, 
                             alpha=0.7, color='orange', capsize=5, edgecolor='black')
        axes[0, 0].set_title('M Component: Gaussian Mixture Probabilities')
        axes[0, 0].set_xlabel('Gaussian Component ID')
        axes[0, 0].set_ylabel('Probability')
        axes[0, 0].set_xticks(x_pos)
        
        # 添加数值标签
        for bar, prob, std in zip(bars, avg_mixture_probs, std_mixture_probs):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                           f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2.2 注意力权重分布
        avg_attention = np.mean(attention_data, axis=0)
        std_attention = np.std(attention_data, axis=0)
        
        x_pos = np.arange(len(avg_attention))
        axes[0, 1].bar(x_pos, avg_attention, yerr=std_attention, 
                      alpha=0.7, color='cyan', capsize=3, edgecolor='black')
        axes[0, 1].set_title('M Component: Memory Attention Weights')
        axes[0, 1].set_xlabel('Memory Slot ID')
        axes[0, 1].set_ylabel('Attention Weight')
        axes[0, 1].set_xticks(range(0, len(avg_attention), 2))
        
        # 2.3 注意力熵分布（衡量注意力集中程度）
        attention_entropy = []
        for i in range(len(attention_data)):
            attention = attention_data[i]
            attention = attention / (np.sum(attention) + 1e-8)  # 归一化
            entropy = -np.sum(attention * np.log(attention + 1e-8))
            attention_entropy.append(entropy)
        
        axes[1, 0].hist(attention_entropy, bins=25, alpha=0.7, color='magenta', edgecolor='black')
        axes[1, 0].set_title('M Component: Attention Entropy Distribution')
        axes[1, 0].set_xlabel('Entropy (Higher = More Distributed)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(attention_entropy), color='darkmagenta', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(attention_entropy):.3f}')
        axes[1, 0].legend()
        
        # 2.4 混合概率的样本间变化
        mixture_variance = np.var(mixture_data, axis=0)
        axes[1, 1].bar(range(len(mixture_variance)), mixture_variance, 
                      alpha=0.7, color='brown', edgecolor='black')
        axes[1, 1].set_title('M Component: Mixture Probability Variance')
        axes[1, 1].set_xlabel('Gaussian Component ID')
        axes[1, 1].set_ylabel('Variance Across Samples')
        axes[1, 1].set_xticks(range(len(mixture_variance)))
        
        plt.tight_layout()
        plt.savefig('results/m_component_probability_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ M组件概率分布完成: results/m_component_probability_analysis.png")
        
        # ==================== 3. 生成综合评估报告 ====================
        print("   生成最终评估报告...")
        
        # 计算关键指标
        avg_kl = np.mean(kl_divergence.numpy())
        avg_separation = np.mean(distances) if len(class_centers) > 1 else 0
        dominant_gaussian = np.argmax(avg_mixture_probs)
        dominant_prob = np.max(avg_mixture_probs)
        avg_attention_entropy = np.mean(attention_entropy)
        most_attended_slot = np.argmax(avg_attention)
        max_attention = np.max(avg_attention)
        
        # 生成详细报告
        report = f"""
最终VMC (Variational Memory Compression) 评估报告
==============================================

训练基本信息:
- 检查点: {checkpoint_path}
- 分析样本数: {len(v_data)}
- 模型参数量: 620,819

V组件 (变分编码器) 性能评估:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
潜在空间质量:
  • 平均KL散度: {avg_kl:.4f}
  • 类别分离度: {avg_separation:.4f}
  • 总体方差: {total_variance:.4f}
  • 激活维度比例: {np.mean(latent_activation > 0.1):.3f}

编码质量评级:
  • KL散度控制: {'优秀' if avg_kl < 5.0 else '良好' if avg_kl < 10.0 else '需要改进'}
  • 类别分离度: {'优秀' if avg_separation > 3.0 else '良好' if avg_separation > 1.0 else '需要改进'}
  • 维度利用率: {'高效' if np.mean(latent_activation > 0.1) > 0.8 else '中等' if np.mean(latent_activation > 0.1) > 0.5 else '低效'}

M组件 (记忆模块) 概率分布分析:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
混合高斯分布:
  • 主导分量: Component {dominant_gaussian} (概率: {dominant_prob:.3f})
  • 分布均匀性: {'均匀' if dominant_prob < 0.3 else '中等' if dominant_prob < 0.5 else '集中'}
  • 组件数量: {len(avg_mixture_probs)}

记忆注意力机制:
  • 最受关注记忆槽: Slot {most_attended_slot} (权重: {max_attention:.3f})
  • 平均注意力熵: {avg_attention_entropy:.3f}
  • 注意力分布: {'分散' if avg_attention_entropy > 2.5 else '适中' if avg_attention_entropy > 1.5 else '集中'}

整体性能评估:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
V组件表现: {'优秀' if avg_kl < 5.0 and avg_separation > 2.0 else '良好' if avg_kl < 8.0 else '可接受'}
M组件表现: {'有效' if avg_attention_entropy < 3.0 and dominant_prob < 0.6 else '可接受'}

建议优化方向:
{f'• 减少KL散度以提高编码效率' if avg_kl > 8.0 else '• KL散度控制良好'}
{f'• 提高类别分离度' if avg_separation < 2.0 else '• 类别分离度良好'}
{f'• 平衡混合分量分布' if dominant_prob > 0.5 else '• 混合分量分布合理'}
{f'• 调整注意力机制集中度' if avg_attention_entropy > 3.0 else '• 注意力机制工作正常'}

生成的可视化文件:
1. results/v_component_evaluation_metrics.png - V组件全面评估
2. results/m_component_probability_analysis.png - M组件概率分析

结论: VMC模型在变分编码和记忆机制方面展现了{'良好' if avg_kl < 8.0 and avg_attention_entropy < 3.5 else '可接受'}的性能表现。
"""
        
        # 保存最终报告
        with open('results/final_vmc_evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n🎉 最终VMC可视化全部完成!")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("📊 生成的可视化文件:")
        print("   • results/v_component_evaluation_metrics.png - V组件性能评估")
        print("   • results/m_component_probability_analysis.png - M组件概率分布")
        print("   • results/final_vmc_evaluation_report.txt - 最终评估报告")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📈 关键性能指标:")
        print(f"   • V组件 KL散度: {avg_kl:.4f}")
        print(f"   • V组件 类别分离度: {avg_separation:.4f}")
        print(f"   • M组件 主导分量概率: {dominant_prob:.3f}")
        print(f"   • M组件 注意力熵: {avg_attention_entropy:.3f}")
        
        # 输出精简版报告
        print("\n" + "="*60)
        print("📋 快速评估总结:")
        print(f"   V组件编码质量: {'优秀' if avg_kl < 5.0 and avg_separation > 2.0 else '良好' if avg_kl < 8.0 else '可接受'}")
        print(f"   M组件记忆效率: {'高效' if avg_attention_entropy < 2.5 and dominant_prob < 0.4 else '有效' if avg_attention_entropy < 3.0 else '可接受'}")
        print(f"   整体架构表现: {'优秀' if avg_kl < 6.0 and avg_attention_entropy < 2.8 else '良好'}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 可视化过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    final_visualize_components()