#!/usr/bin/env python3
"""
简化的VMC可视化脚本 - 避免segmentation fault问题
"""
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from config import VMCConfig
from vmc_model import create_vmc_model
from dataset import VMCDataset

def simple_visualize_components():
    """简化的VMC组件可视化"""
    
    print("🎨 简化VMC可视化开始...")
    
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
                
                # 将变分编码转换为记忆查询 (64 -> 32维)
                memory_query = model.var_to_memory(v_output)
                
                # M组件 - 记忆模块  
                memory_output, attention_weights, mixture_probs = model.memory_module(memory_query)
                
                # C组件 - 控制器
                c_output, gate_weight = model.controller(v_output, memory_output)
                
                break  # 只处理第一个批次
        
        # 转换为numpy数组以便可视化
        v_data = v_output.cpu().numpy()
        m_data = memory_output.cpu().numpy() 
        c_data = c_output.cpu().numpy()
        labels_np = labels.cpu().numpy()
        attention_data = attention_weights.cpu().numpy()
        mixture_data = mixture_probs.cpu().numpy()
        gate_data = gate_weight.cpu().numpy()
        
        print("📊 开始生成可视化图表...")
        
        # 1. V组件可视化 - 使用PCA降维
        print("   生成V组件可视化...")
        plt.figure(figsize=(12, 4))
        
        # V组件的潜在空间分布
        plt.subplot(1, 3, 1)
        pca_v = PCA(n_components=2)
        v_2d = pca_v.fit_transform(v_data)
        scatter = plt.scatter(v_2d[:, 0], v_2d[:, 1], c=labels_np, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('V组件: 变分潜在空间 (PCA)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        # V组件的均值分布
        plt.subplot(1, 3, 2)
        mu_np = mu.cpu().numpy()
        plt.hist(mu_np.flatten(), bins=50, alpha=0.7, color='blue')
        plt.title('V组件: 潜在均值分布')
        plt.xlabel('值')
        plt.ylabel('频率')
        
        # V组件的方差分布
        plt.subplot(1, 3, 3)
        logvar_np = logvar.cpu().numpy()
        plt.hist(logvar_np.flatten(), bins=50, alpha=0.7, color='red')
        plt.title('V组件: 对数方差分布')
        plt.xlabel('log(σ²)')
        plt.ylabel('频率')
        
        plt.tight_layout()
        plt.savefig('results/v_component_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ V组件可视化完成: results/v_component_visualization.png")
        
        # 2. M组件可视化 - 记忆模块
        print("   生成M组件可视化...")
        plt.figure(figsize=(12, 4))
        
        # 记忆输出的PCA
        plt.subplot(1, 3, 1)
        pca_m = PCA(n_components=2)
        m_2d = pca_m.fit_transform(m_data)
        scatter = plt.scatter(m_2d[:, 0], m_2d[:, 1], c=labels_np, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('M组件: 记忆输出空间 (PCA)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        # 记忆模块的键值分布
        plt.subplot(1, 3, 2)
        memory_keys = model.memory_module.memory_keys.detach().cpu().numpy()
        plt.imshow(memory_keys, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('M组件: 记忆键矩阵')
        plt.xlabel('记忆维度')
        plt.ylabel('记忆槽')
        
        # 记忆输出分布
        plt.subplot(1, 3, 3)
        plt.hist(m_data.flatten(), bins=50, alpha=0.7, color='green')
        plt.title('M组件: 记忆输出分布')
        plt.xlabel('值')
        plt.ylabel('频率')
        
        plt.tight_layout()
        plt.savefig('results/m_component_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ M组件可视化完成: results/m_component_visualization.png")
        
        # 3. C组件可视化 - 控制器
        print("   生成C组件可视化...")
        plt.figure(figsize=(12, 4))
        
        # 控制器输出的概率分布
        plt.subplot(1, 3, 1)
        c_probs = torch.softmax(torch.tensor(c_data), dim=1).numpy()
        for i in range(10):  # MNIST有10个类
            class_probs = c_probs[labels_np == i].mean(axis=0)
            plt.bar(range(10), class_probs, alpha=0.7, label=f'真实类{i}')
        plt.title('C组件: 各类别预测概率')
        plt.xlabel('预测类别')
        plt.ylabel('平均概率') 
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 预测准确性
        plt.subplot(1, 3, 2)
        predictions = np.argmax(c_data, axis=1)
        accuracy_per_class = []
        for i in range(10):
            mask = labels_np == i
            if mask.sum() > 0:
                acc = (predictions[mask] == i).mean()
                accuracy_per_class.append(acc)
            else:
                accuracy_per_class.append(0)
        
        plt.bar(range(10), accuracy_per_class, color='orange', alpha=0.7)
        plt.title('C组件: 各类别准确率')
        plt.xlabel('类别')
        plt.ylabel('准确率')
        plt.ylim(0, 1)
        
        # 混淆矩阵可视化
        plt.subplot(1, 3, 3)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels_np, predictions)
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.title('C组件: 混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        
        plt.tight_layout()
        plt.savefig('results/c_component_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ C组件可视化完成: results/c_component_visualization.png")
        
        # 4. 生成综合报告
        print("   生成综合性能报告...")
        overall_accuracy = (predictions == labels_np).mean()
        
        plt.figure(figsize=(10, 6))
        
        # 性能指标
        plt.subplot(2, 2, 1)
        metrics = ['整体准确率', '平均类别准确率', 'V组件方差', 'M组件激活度']
        values = [
            overall_accuracy,
            np.mean(accuracy_per_class),
            np.exp(logvar_np).mean(),  # 平均方差
            (m_data > 0).mean()  # 记忆激活比例
        ]
        plt.bar(range(len(metrics)), values, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
        plt.xticks(range(len(metrics)), metrics, rotation=45)
        plt.title('VMC整体性能指标')
        plt.ylabel('数值')
        
        # 损失组成
        plt.subplot(2, 2, 2)
        epoch_info = checkpoint.get('epoch_info', {})
        loss_components = ['分类损失', 'KL损失', '记忆损失']
        loss_values = [
            epoch_info.get('classification_loss', 0),
            epoch_info.get('kl_loss', 0), 
            epoch_info.get('memory_loss', 0)
        ]
        plt.pie(loss_values, labels=loss_components, autopct='%1.1f%%', startangle=90)
        plt.title('损失函数组成')
        
        # 各组件维度分析
        plt.subplot(2, 2, 3)
        component_dims = ['V输出', 'M输出', 'C输出']
        dims = [v_data.shape[1], m_data.shape[1], c_data.shape[1]]
        plt.bar(component_dims, dims, color=['purple', 'cyan', 'yellow'], alpha=0.7)
        plt.title('各组件输出维度')
        plt.ylabel('维度数')
        
        # 训练历史(如果有)
        plt.subplot(2, 2, 4)
        if 'train_history' in checkpoint:
            history = checkpoint['train_history']
            epochs = range(1, len(history) + 1)
            plt.plot(epochs, [h['train_acc'] for h in history], 'b-', label='训练准确率')
            plt.plot(epochs, [h['test_acc'] for h in history], 'r-', label='测试准确率')
            plt.xlabel('Epoch')
            plt.ylabel('准确率')
            plt.legend()
            plt.title('训练历史')
        else:
            plt.text(0.5, 0.5, '训练历史不可用', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('训练历史')
        
        plt.tight_layout()
        plt.savefig('results/vmc_comprehensive_report.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ 综合报告完成: results/vmc_comprehensive_report.png")
        
        print("\n🎉 VMC可视化全部完成!")
        print("📊 生成的可视化文件:")
        print("   • results/v_component_visualization.png - V组件(变分编码器)分析")
        print("   • results/m_component_visualization.png - M组件(记忆模块)分析") 
        print("   • results/c_component_visualization.png - C组件(控制器)分析")
        print("   • results/vmc_comprehensive_report.png - VMC综合性能报告")
        print(f"\n📈 整体性能: 准确率 {overall_accuracy:.2%}")
        
    except Exception as e:
        print(f"❌ 可视化过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_visualize_components()