"""
VMC Demo 主入口文件
Variational Memory Compression 演示项目
"""
import argparse
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端避免segmentation fault
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from datetime import datetime

from config import VMCConfig
from trainer import VMCTrainer, quick_train
from vmc_model import create_vmc_model, test_vmc_model
from dataset import VMCDataset, test_dataset
from visualizer import VMCVisualizer

def create_demo_visualizations(model, test_loader):
    """创建演示可视化 - 修复版本，避免segmentation fault"""
    print("   📊 生成V组件和M组件可视化...")
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    try:
        model.eval()
        
        # 收集数据进行分析 (只处理前几个批次)
        all_v_outputs = []
        all_mu = []
        all_logvar = []
        all_attention_weights = []
        all_mixture_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                if batch_idx >= 3:  # 只处理前3个批次用于演示
                    break
                    
                images = images.to(VMCConfig.device)
                labels = labels.to(VMCConfig.device)
                
                # 扁平化图像
                images_flat = images.view(images.size(0), -1)
                
                # V组件 - 变分编码器
                v_output, mu, logvar = model.variational_encoder(images_flat)
                
                # 将变分编码转换为记忆查询
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
        
        # 合并数据
        v_data = torch.cat(all_v_outputs, dim=0).numpy()
        mu_data = torch.cat(all_mu, dim=0).numpy()
        logvar_data = torch.cat(all_logvar, dim=0).numpy()
        attention_data = torch.cat(all_attention_weights, dim=0).numpy()
        mixture_data = torch.cat(all_mixture_probs, dim=0).numpy()
        labels_data = torch.cat(all_labels, dim=0).numpy()
        
        # 1. V组件评估
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # V组件潜在空间分布
        pca = PCA(n_components=2)
        v_2d = pca.fit_transform(v_data)
        scatter = axes[0].scatter(v_2d[:, 0], v_2d[:, 1], c=labels_data, cmap='tab10', alpha=0.6, s=20)
        axes[0].set_title('V Component: Latent Space (PCA)')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        plt.colorbar(scatter, ax=axes[0])
        
        # KL散度分布
        kl_divergence = -0.5 * torch.sum(1 + torch.tensor(logvar_data) - torch.tensor(mu_data).pow(2) - torch.tensor(logvar_data).exp(), dim=1)
        axes[1].hist(kl_divergence.numpy(), bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[1].set_title('V Component: KL Divergence')
        axes[1].set_xlabel('KL Divergence')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(np.mean(kl_divergence.numpy()), color='darkred', linestyle='--', linewidth=2)
        
        # 潜在维度激活度
        latent_activation = np.mean(np.abs(v_data), axis=0)
        axes[2].bar(range(len(latent_activation)), latent_activation, alpha=0.7, color='blue')
        axes[2].set_title('V Component: Dimension Activation')
        axes[2].set_xlabel('Latent Dimension')
        axes[2].set_ylabel('Avg Activation')
        axes[2].set_xticks(range(0, len(latent_activation), 8))
        
        plt.tight_layout()
        plt.savefig('results/demo_v_component_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. M组件概率分布
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 混合高斯分量概率
        avg_mixture_probs = np.mean(mixture_data, axis=0)
        std_mixture_probs = np.std(mixture_data, axis=0)
        
        x_pos = np.arange(len(avg_mixture_probs))
        bars = axes[0].bar(x_pos, avg_mixture_probs, yerr=std_mixture_probs, 
                          alpha=0.7, color='orange', capsize=5, edgecolor='black')
        axes[0].set_title('M Component: Gaussian Mixture Probabilities')
        axes[0].set_xlabel('Component ID')
        axes[0].set_ylabel('Probability')
        axes[0].set_xticks(x_pos)
        
        # 注意力权重分布
        avg_attention = np.mean(attention_data, axis=0)
        axes[1].bar(range(len(avg_attention)), avg_attention, alpha=0.7, color='cyan', edgecolor='black')
        axes[1].set_title('M Component: Memory Attention Weights')
        axes[1].set_xlabel('Memory Slot ID')
        axes[1].set_ylabel('Attention Weight')
        axes[1].set_xticks(range(0, len(avg_attention), 2))
        
        plt.tight_layout()
        plt.savefig('results/demo_m_component_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. 生成简化报告
        avg_kl = np.mean(kl_divergence.numpy())
        dominant_gaussian = np.argmax(avg_mixture_probs)
        dominant_prob = np.max(avg_mixture_probs)
        
        # 计算注意力熵
        attention_entropy = []
        for i in range(len(attention_data)):
            attention = attention_data[i]
            attention = attention / (np.sum(attention) + 1e-8)
            entropy = -np.sum(attention * np.log(attention + 1e-8))
            attention_entropy.append(entropy)
        avg_attention_entropy = np.mean(attention_entropy)
        
        report = f"""
VMC 演示结果报告
===============

V组件 (变分编码器) 性能:
- 平均KL散度: {avg_kl:.4f}
- 潜在维度: {len(latent_activation)}
- 维度利用率: {np.mean(latent_activation > 0.1):.3f}

M组件 (记忆模块) 性能:
- 主导分量: Component {dominant_gaussian} (概率: {dominant_prob:.3f})
- 平均注意力熵: {avg_attention_entropy:.3f}
- 记忆槽数量: {len(avg_attention)}

生成的可视化文件:
- results/demo_v_component_analysis.png
- results/demo_m_component_analysis.png

总体评估: {'优秀' if avg_kl < 6.0 and avg_attention_entropy < 3.0 else '良好'}
"""
        
        with open('results/demo_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("   ✅ V组件分析完成: results/demo_v_component_analysis.png")
        print("   ✅ M组件分析完成: results/demo_m_component_analysis.png")
        print("   ✅ 演示报告完成: results/demo_report.txt")
        print(f"   📊 关键指标: KL散度={avg_kl:.3f}, 注意力熵={avg_attention_entropy:.3f}")
        
    except Exception as e:
        print(f"   ❌ 可视化生成失败: {e}")
        import traceback
        traceback.print_exc()

def print_banner():
    """打印项目横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    VMC Demo Project                                   ║
    ║              Variational Memory Compression                           ║
    ║                                                                      ║
    ║    V (Variational) → M (Memory) → C (Controller)                     ║
    ║                                                                      ║
    ║           针对 MacBook Air M1 16GB 内存优化                           ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def train_mode():
    """训练模式"""
    print("🚀 启动VMC训练模式")
    print("="*60)
    
    # 显示配置
    VMCConfig.print_config()
    
    # 询问是否修改配置
    if input("\n是否修改训练配置？(y/n): ").lower() == 'y':
        print("\n⚙️ 配置修改选项:")
        print("1. 快速训练 (10 epochs, 适合演示)")
        print("2. 标准训练 (30 epochs, 平衡效果和时间)")
        print("3. 完整训练 (50 epochs, 最佳效果)")
        print("4. 自定义配置")
        
        choice = input("请选择 (1-4): ").strip()
        
        if choice == '1':
            VMCConfig.epochs = 10
            VMCConfig.save_interval = 2
            print("✅ 设置为快速训练模式")
        elif choice == '2':
            VMCConfig.epochs = 30
            VMCConfig.save_interval = 5
            print("✅ 设置为标准训练模式")
        elif choice == '3':
            VMCConfig.epochs = 50
            VMCConfig.save_interval = 10
            print("✅ 设置为完整训练模式")
        elif choice == '4':
            try:
                epochs = int(input("训练轮数 (默认30): ") or "30")
                batch_size = int(input("批次大小 (默认32): ") or "32")
                learning_rate = float(input("学习率 (默认1e-3): ") or "1e-3")
                
                VMCConfig.epochs = epochs
                VMCConfig.batch_size = batch_size
                VMCConfig.learning_rate = learning_rate
                VMCConfig.save_interval = max(1, epochs // 6)
                
                print("✅ 自定义配置已设置")
            except ValueError:
                print("❌ 输入格式错误，使用默认配置")
    
    try:
        # 创建训练器
        print("\n🏗️ 创建VMC训练器...")
        trainer = VMCTrainer()
        
        # 开始训练
        print("\n🎯 开始训练...")
        results = trainer.train()
        
        if results:
            print(f"\n🎉 训练成功完成!")
            print(f"   训练时间: {results['training_time']}")
            print(f"   最终训练准确率: {results['final_train_accuracy']:.2f}%")
            print(f"   最终测试准确率: {results['final_test_accuracy']:.2f}%")
            print(f"   模型保存位置: {results['checkpoint_path']}")
            
            # 询问是否生成详细报告
            if input("\n是否生成详细训练报告？(y/n): ").lower() == 'y':
                generate_training_report(trainer, results)
        else:
            print("❌ 训练失败")
            
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")

def visualize_mode():
    """可视化模式"""
    print("🎨 启动VMC可视化模式")
    print("="*60)
    
    # 检查是否有已训练的模型
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("❌ 未找到训练检查点目录")
        print("请先运行训练模式: python main.py --mode train")
        return
    
    # 列出可用的检查点
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if not checkpoints:
        print("❌ 未找到训练检查点文件")
        print("请先运行训练模式: python main.py --mode train")
        return
    
    print("📂 可用的训练检查点:")
    for i, checkpoint in enumerate(checkpoints):
        print(f"   {i+1}. {checkpoint}")
    
    try:
        choice = int(input(f"请选择检查点 (1-{len(checkpoints)}): ")) - 1
        if 0 <= choice < len(checkpoints):
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[choice])
        else:
            print("❌ 无效选择，使用最新的检查点")
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
    except ValueError:
        print("❌ 输入无效，使用最新的检查点")
        checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
    
    try:
        # 加载检查点
        print(f"📂 加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=VMCConfig.device, weights_only=False)
        
        # 创建模型并加载权重
        model = create_vmc_model(VMCConfig())
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建数据集
        dataset_manager = VMCDataset()
        train_loader, test_loader = dataset_manager.create_dataloaders()
        
        # 创建可视化器
        visualizer = VMCVisualizer(model, VMCConfig())
        
        epoch = checkpoint['epoch']
        
        # 可视化选项
        while True:
            print(f"\n🎨 VMC可视化选项 (基于Epoch {epoch}):")
            print("1. V组件 (Variational Encoder) 可视化")
            print("2. M组件 (Memory Module) 可视化")
            print("3. C组件 (Controller) 可视化")
            print("4. 完整流水线可视化")
            print("5. 生成所有可视化")
            print("6. 数据集样本可视化")
            print("0. 返回主菜单")
            
            choice = input("请选择 (0-6): ").strip()
            
            if choice == '1':
                print("📊 生成V组件可视化...")
                visualizer.visualize_variational_component(test_loader, epoch)
            elif choice == '2':
                print("🧠 生成M组件可视化...")
                visualizer.visualize_memory_component(test_loader, epoch)
            elif choice == '3':
                print("🎮 生成C组件可视化...")
                visualizer.visualize_controller_component(test_loader, epoch)
            elif choice == '4':
                print("🔄 生成完整流水线可视化...")
                visualizer.visualize_complete_pipeline(test_loader, epoch)
            elif choice == '5':
                print("🎨 生成所有可视化...")
                visualizer.visualize_variational_component(test_loader, epoch)
                visualizer.visualize_memory_component(test_loader, epoch)
                visualizer.visualize_controller_component(test_loader, epoch)
                visualizer.visualize_complete_pipeline(test_loader, epoch)
                print("✅ 所有可视化完成!")
            elif choice == '6':
                print("📊 生成数据集样本可视化...")
                dataset_manager.visualize_samples(train_loader)
            elif choice == '0':
                break
            else:
                print("❌ 无效选择，请重新输入")
        
    except Exception as e:
        print(f"❌ 可视化过程中出错: {e}")

def test_mode():
    """测试模式"""
    print("🧪 启动VMC测试模式")
    print("="*60)
    
    try:
        print("\n1. 测试配置...")
        VMCConfig.print_config()
        
        print("\n2. 测试数据集...")
        test_dataset()
        
        print("\n3. 测试VMC模型...")
        test_vmc_model()
        
        print("\n4. 测试训练器创建...")
        # 创建一个简单的训练器实例
        trainer = VMCTrainer()
        print(f"   ✅ 训练器创建成功")
        print(f"   模型参数: {sum(p.numel() for p in trainer.model.parameters()):,}")
        
        print("\n🎉 所有组件测试通过!")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")

def demo_mode():
    """演示模式 - 快速展示VMC的核心功能"""
    print("🎬 启动VMC演示模式")
    print("="*60)
    
    print("这是一个快速演示，将展示VMC的核心组件:")
    print("✓ V组件: 将输入编码为变分潜在表示")
    print("✓ M组件: 使用混合高斯分布的记忆机制")
    print("✓ C组件: 融合变分和记忆信息进行分类")
    
    if input("\n是否开始演示？(y/n): ").lower() != 'y':
        return
    
    # 设置快速演示参数
    original_epochs = VMCConfig.epochs
    original_save_interval = VMCConfig.save_interval
    
    VMCConfig.epochs = 5  # 只训练5个epoch用于演示
    VMCConfig.save_interval = 2  # 每2个epoch可视化一次
    
    try:
        print("\n🎯 开始快速演示训练...")
        trainer = VMCTrainer()
        
        print("📊 首先查看数据集样本...")
        trainer.dataset_manager.visualize_samples(trainer.train_loader)
        
        print("\n🚀 开始演示训练 (5个epoch)...")
        results = trainer.train(demo_mode=True)
        
        if results:
            print("\n🎉 演示训练完成!")
            print(f"   最终测试准确率: {results['final_test_accuracy']:.2f}%")
            
            print("\n🎨 生成最终演示可视化...")
            # 使用修复后的可视化方法
            create_demo_visualizations(trainer.model, trainer.test_loader)
            
            print("✅ VMC演示完成! 请查看生成的可视化结果。")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
    
    finally:
        # 恢复原始配置
        VMCConfig.epochs = original_epochs
        VMCConfig.save_interval = original_save_interval

def interactive_mode():
    """交互模式"""
    print("🎮 启动VMC交互模式")
    
    while True:
        print("\n" + "="*70)
        print("VMC Demo 主菜单:")
        print("1. 🚀 训练模式 - 训练VMC模型")
        print("2. 🎨 可视化模式 - 查看训练结果")
        print("3. 🎬 演示模式 - 快速演示VMC功能")
        print("4. 🧪 测试模式 - 测试各个组件")
        print("5. ⚙️  查看配置 - 显示当前配置")
        print("6. 📖 显示帮助 - VMC原理和使用说明")
        print("0. 👋 退出程序")
        print("="*70)
        
        choice = input("请选择操作 (0-6): ").strip()
        
        if choice == '1':
            train_mode()
        elif choice == '2':
            visualize_mode()
        elif choice == '3':
            demo_mode()
        elif choice == '4':
            test_mode()
        elif choice == '5':
            VMCConfig.print_config()
        elif choice == '6':
            show_help()
        elif choice == '0':
            print("👋 感谢使用VMC Demo! 再见!")
            break
        else:
            print("❌ 无效选择，请重新输入")

def show_help():
    """显示帮助信息"""
    help_text = """
    📖 VMC (Variational Memory Compression) 原理和使用说明
    
    🔬 VMC 原理:
    VMC是一个将变分自编码器和记忆机制结合的深度学习架构:
    
    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
    │  V Component │───▶│  M Component │───▶│ C Component │
    │  (Variational)│    │   (Memory)   │    │ (Controller)│
    └─────────────┘    └──────────────┘    └─────────────┘
    
    • V组件: 将输入数据编码为变分潜在表示，捕获数据的概率分布
    • M组件: 使用混合高斯分布的记忆机制，存储和检索有用的模式
    • C组件: 融合变分编码和记忆信息，进行最终的分类决策
    
    🎯 主要特性:
    • 变分推理: 学习数据的潜在分布而不是点估计
    • 记忆机制: 通过注意力机制访问外部记忆
    • 混合高斯: 灵活的概率分布建模
    • 端到端训练: 三个组件协同优化
    
    💻 使用建议:
    • 演示模式: 快速了解VMC功能 (5分钟)
    • 训练模式: 完整训练VMC模型 (30-60分钟)
    • 可视化模式: 深入分析训练结果
    • 测试模式: 验证各组件功能
    
    🔧 针对MacBook Air M1 16GB优化:
    • 批次大小: 32 (平衡性能和内存)
    • 记忆槽: 16个 (适中的记忆容量)
    • 混合组件: 8个高斯分量
    • 使用MPS加速训练
    """
    print(help_text)

def generate_training_report(trainer, results):
    """生成详细的训练报告"""
    print("\n📋 生成训练报告...")
    
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"vmc_training_report_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("VMC Training Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 训练配置
        f.write("Training Configuration:\n")
        f.write(f"  Epochs: {VMCConfig.epochs}\n")
        f.write(f"  Batch Size: {VMCConfig.batch_size}\n")
        f.write(f"  Learning Rate: {VMCConfig.learning_rate}\n")
        f.write(f"  Variational Dim: {VMCConfig.variational_dim}\n")
        f.write(f"  Memory Size: {VMCConfig.memory_size}\n")
        f.write(f"  Memory Dim: {VMCConfig.memory_dim}\n")
        f.write(f"  Num Gaussians: {VMCConfig.num_gaussians}\n\n")
        
        # 训练结果
        f.write("Training Results:\n")
        f.write(f"  Training Time: {results['training_time']}\n")
        f.write(f"  Final Train Accuracy: {results['final_train_accuracy']:.2f}%\n")
        f.write(f"  Final Test Accuracy: {results['final_test_accuracy']:.2f}%\n")
        f.write(f"  Best Test Accuracy: {max(trainer.test_accuracies):.2f}%\n\n")
        
        # 组件统计
        if trainer.component_stats['variational']:
            f.write("Component Statistics (Final):\n")
            final_v = trainer.component_stats['variational'][-1]
            final_m = trainer.component_stats['memory'][-1]
            final_c = trainer.component_stats['controller'][-1]
            
            f.write(f"  V Component:\n")
            f.write(f"    Mean KL Divergence: {final_v['mean_kl_div']:.4f}\n")
            f.write(f"    Mean Std: {final_v['mean_std']:.4f}\n")
            f.write(f"    Encoding Quality: {final_v['encoding_quality']:.4f}\n")
            
            f.write(f"  M Component:\n")
            f.write(f"    Attention Entropy: {final_m['mean_attention_entropy']:.4f}\n")
            f.write(f"    Mixture Diversity: {final_m['mixture_diversity']:.4f}\n")
            f.write(f"    Query-Retrieval Similarity: {final_m['query_retrieval_similarity']:.4f}\n")
            
            f.write(f"  C Component:\n")
            f.write(f"    Accuracy: {final_c['accuracy']:.4f}\n")
            f.write(f"    Mean Gate Weight: {final_c['mean_gate_weight']:.4f}\n")
            f.write(f"    Mean Confidence: {final_c['mean_confidence']:.4f}\n")
    
    print(f"📋 训练报告已保存: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VMC Demo Project")
    parser.add_argument(
        '--mode', 
        choices=['train', 'visualize', 'test', 'demo', 'interactive'],
        default='interactive',
        help='运行模式'
    )
    parser.add_argument(
        '--config',
        action='store_true',
        help='显示配置信息'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速训练模式'
    )
    
    args = parser.parse_args()
    
    # 打印横幅
    print_banner()
    
    # 显示配置
    if args.config:
        VMCConfig.print_config()
        return
    
    # 快速训练
    if args.quick:
        VMCConfig.epochs = 10
        VMCConfig.save_interval = 2
        quick_train()
        return
    
    # 根据模式运行
    if args.mode == 'train':
        train_mode()
    elif args.mode == 'visualize':
        visualize_mode()
    elif args.mode == 'test':
        test_mode()
    elif args.mode == 'demo':
        demo_mode()
    elif args.mode == 'interactive':
        interactive_mode()

if __name__ == "__main__":
    main()