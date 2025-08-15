"""
VMC可视化模块 - 详细展示V、M、C三个组件的训练过程和结果
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免segmentation fault
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from config import VMCConfig

class VMCVisualizer:
    """VMC各组件可视化器"""
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config if config else VMCConfig()
        self.model.eval()
        
        # 设置绘图风格
        plt.style.use('default')
        sns.set_palette("husl")
        
        print("🎨 VMC可视化器初始化完成")
    
    def visualize_variational_component(self, data_loader, epoch, save_dir="results"):
        """可视化V组件 - 变分编码器的结果"""
        print(f"📊 可视化V组件 (Variational Encoder) - Epoch {epoch}")
        
        # 收集变分编码数据
        var_codes = []
        var_mus = []
        var_logvars = []
        labels = []
        
        with torch.no_grad():
            for i, (x, y) in enumerate(data_loader):
                if i >= 20:  # 只使用前20个batch避免内存问题
                    break
                
                x = x.view(x.size(0), -1).to(self.config.device)
                y = y.to(self.config.device)
                
                # 获取变分编码
                var_code, var_mu, var_logvar = self.model.variational_encoder(x)
                
                var_codes.append(var_code.cpu())
                var_mus.append(var_mu.cpu())
                var_logvars.append(var_logvar.cpu())
                labels.append(y.cpu())
        
        # 合并数据
        var_codes = torch.cat(var_codes, dim=0).numpy()
        var_mus = torch.cat(var_mus, dim=0).numpy()
        var_logvars = torch.cat(var_logvars, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'V Component (Variational Encoder) - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # 1. 变分编码的t-SNE可视化
        if var_codes.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            var_codes_2d = tsne.fit_transform(var_codes)
        else:
            var_codes_2d = var_codes
        
        scatter = axes[0, 0].scatter(var_codes_2d[:, 0], var_codes_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        axes[0, 0].set_title('Variational Codes (t-SNE)')
        axes[0, 0].set_xlabel('Dimension 1')
        axes[0, 0].set_ylabel('Dimension 2')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. 均值的分布
        if var_mus.shape[1] > 2:
            pca = PCA(n_components=2)
            var_mus_2d = pca.fit_transform(var_mus)
        else:
            var_mus_2d = var_mus
        
        scatter2 = axes[0, 1].scatter(var_mus_2d[:, 0], var_mus_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        axes[0, 1].set_title('Variational Means μ (PCA)')
        axes[0, 1].set_xlabel('PC1')
        axes[0, 1].set_ylabel('PC2')
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        # 3. 方差的分布
        var_stds = np.exp(0.5 * var_logvars)
        axes[0, 2].hist(var_stds.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 2].set_title('Variational Standard Deviations σ')
        axes[0, 2].set_xlabel('σ value')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(var_stds.mean(), color='red', linestyle='--', label=f'Mean: {var_stds.mean():.3f}')
        axes[0, 2].legend()
        
        # 4. KL散度分布
        kl_divs = -0.5 * np.sum(1 + var_logvars - var_mus**2 - np.exp(var_logvars), axis=1)
        axes[1, 0].hist(kl_divs, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_title('KL Divergence Distribution')
        axes[1, 0].set_xlabel('KL Divergence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(kl_divs.mean(), color='red', linestyle='--', label=f'Mean: {kl_divs.mean():.3f}')
        axes[1, 0].legend()
        
        # 5. 各维度方差热图
        var_mean_by_class = []
        for class_id in range(10):
            mask = labels == class_id
            if np.sum(mask) > 0:
                class_var = var_stds[mask].mean(axis=0)
                var_mean_by_class.append(class_var)
        
        var_heatmap = np.array(var_mean_by_class)
        im = axes[1, 1].imshow(var_heatmap, cmap='viridis', aspect='auto')
        axes[1, 1].set_title('Mean σ by Class and Dimension')
        axes[1, 1].set_xlabel('Latent Dimension')
        axes[1, 1].set_ylabel('Class')
        axes[1, 1].set_yticks(range(10))
        plt.colorbar(im, ax=axes[1, 1])
        
        # 6. 编码质量评估（重建能力）
        reconstruction_quality = []
        sample_indices = np.random.choice(len(var_codes), min(100, len(var_codes)), replace=False)
        
        for idx in sample_indices:
            # 计算编码质量的代理指标
            mu = var_mus[idx]
            logvar = var_logvars[idx]
            # 使用信息量作为质量指标
            info_content = -0.5 * np.sum(logvar)
            reconstruction_quality.append(info_content)
        
        axes[1, 2].scatter(range(len(reconstruction_quality)), reconstruction_quality, alpha=0.7, color='green')
        axes[1, 2].set_title('Encoding Quality (Information Content)')
        axes[1, 2].set_xlabel('Sample Index')
        axes[1, 2].set_ylabel('Information Content')
        
        plt.tight_layout()
        
        # 保存结果
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'variational_analysis_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 V组件可视化已保存: {save_path}")
        
        # 返回统计信息
        stats = {
            'mean_kl_div': float(kl_divs.mean()),
            'mean_std': float(var_stds.mean()),
            'encoding_quality': float(np.mean(reconstruction_quality))
        }
        
        return stats
    
    def visualize_memory_component(self, data_loader, epoch, save_dir="results"):
        """可视化M组件 - 记忆模块和混合高斯分布"""
        print(f"🧠 可视化M组件 (Memory Module) - Epoch {epoch}")
        
        # 收集记忆相关数据
        attention_weights_all = []
        mixture_probs_all = []
        memory_queries = []
        retrieved_memories = []
        labels = []
        
        with torch.no_grad():
            for i, (x, y) in enumerate(data_loader):
                if i >= 15:  # 限制批次数量
                    break
                
                x = x.view(x.size(0), -1).to(self.config.device)
                y = y.to(self.config.device)
                
                # 获取记忆相关输出
                _, components = self.model(x)
                
                attention_weights_all.append(components['memory']['attention_weights'].cpu())
                mixture_probs_all.append(components['memory']['mixture_probs'].cpu())
                memory_queries.append(components['memory']['query'].cpu())
                retrieved_memories.append(components['memory']['retrieved'].cpu())
                labels.append(y.cpu())
        
        # 合并数据
        attention_weights = torch.cat(attention_weights_all, dim=0).numpy()
        mixture_probs = torch.cat(mixture_probs_all, dim=0).numpy()
        memory_queries = torch.cat(memory_queries, dim=0).numpy()
        retrieved_memories = torch.cat(retrieved_memories, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()
        
        # 创建可视化
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle(f'M Component (Memory Module) - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # 1. 注意力权重热图
        mean_attention_by_class = []
        for class_id in range(10):
            mask = labels == class_id
            if np.sum(mask) > 0:
                class_attention = attention_weights[mask].mean(axis=0)
                mean_attention_by_class.append(class_attention)
        
        attention_heatmap = np.array(mean_attention_by_class)
        im1 = axes[0, 0].imshow(attention_heatmap, cmap='Blues', aspect='auto')
        axes[0, 0].set_title('Mean Attention Weights by Class')
        axes[0, 0].set_xlabel('Memory Slot')
        axes[0, 0].set_ylabel('Class')
        axes[0, 0].set_yticks(range(10))
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. 混合高斯分布概率
        mean_mixture_by_class = []
        for class_id in range(10):
            mask = labels == class_id
            if np.sum(mask) > 0:
                class_mixture = mixture_probs[mask].mean(axis=0)
                mean_mixture_by_class.append(class_mixture)
        
        mixture_heatmap = np.array(mean_mixture_by_class)
        im2 = axes[0, 1].imshow(mixture_heatmap, cmap='Reds', aspect='auto')
        axes[0, 1].set_title('Mixture Gaussian Probabilities by Class')
        axes[0, 1].set_xlabel('Gaussian Component')
        axes[0, 1].set_ylabel('Class')
        axes[0, 1].set_yticks(range(10))
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. 记忆槽激活统计
        memory_activation = attention_weights.mean(axis=0)
        bars = axes[0, 2].bar(range(len(memory_activation)), memory_activation, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Memory Slot Activation Frequency')
        axes[0, 2].set_xlabel('Memory Slot Index')
        axes[0, 2].set_ylabel('Mean Activation')
        
        # 标注最活跃的记忆槽
        max_idx = np.argmax(memory_activation)
        axes[0, 2].annotate(f'Max: {memory_activation[max_idx]:.3f}', 
                           xy=(max_idx, memory_activation[max_idx]),
                           xytext=(max_idx, memory_activation[max_idx] + 0.01),
                           arrowprops=dict(arrowstyle='->', color='red'))
        
        # 4. 混合高斯分布可视化
        gaussian_means = self.model.memory_module.gaussian_means.detach().cpu().numpy()
        gaussian_logvars = self.model.memory_module.gaussian_logvars.detach().cpu().numpy()
        gaussian_weights = torch.softmax(self.model.memory_module.gaussian_weights, dim=0).detach().cpu().numpy()
        
        # 使用PCA将高维高斯分布投影到2D
        if gaussian_means.shape[1] > 2:
            pca = PCA(n_components=2)
            gaussian_means_2d = pca.fit_transform(gaussian_means)
            # 也投影方差（简化处理）
            gaussian_stds_2d = np.exp(0.5 * pca.transform(gaussian_logvars))
        else:
            gaussian_means_2d = gaussian_means
            gaussian_stds_2d = np.exp(0.5 * gaussian_logvars)
        
        for i in range(self.config.num_gaussians):
            # 绘制高斯分布的椭圆
            from matplotlib.patches import Ellipse
            ell = Ellipse(gaussian_means_2d[i], 
                         width=2*gaussian_stds_2d[i, 0], 
                         height=2*gaussian_stds_2d[i, 1],
                         alpha=0.3 + 0.7 * gaussian_weights[i],
                         color=plt.cm.tab10(i))
            axes[1, 0].add_patch(ell)
            axes[1, 0].scatter(gaussian_means_2d[i, 0], gaussian_means_2d[i, 1], 
                              s=100*gaussian_weights[i], c=[plt.cm.tab10(i)], 
                              edgecolors='black', linewidths=2)
        
        axes[1, 0].set_title('Mixture Gaussian Distribution (2D Projection)')
        axes[1, 0].set_xlabel('Dimension 1')
        axes[1, 0].set_ylabel('Dimension 2')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 高斯分量权重
        axes[1, 1].pie(gaussian_weights, labels=[f'G{i}' for i in range(len(gaussian_weights))], 
                       autopct='%1.2f%%', startangle=90, colors=plt.cm.tab10.colors[:len(gaussian_weights)])
        axes[1, 1].set_title('Gaussian Component Weights')
        
        # 6. 记忆查询vs检索结果相似性
        query_retrieval_sim = []
        for i in range(min(100, len(memory_queries))):
            # 计算查询和检索结果的余弦相似性
            query = memory_queries[i]
            retrieved = retrieved_memories[i]
            similarity = np.dot(query, retrieved) / (np.linalg.norm(query) * np.linalg.norm(retrieved))
            query_retrieval_sim.append(similarity)
        
        axes[1, 2].hist(query_retrieval_sim, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 2].set_title('Query-Retrieval Similarity')
        axes[1, 2].set_xlabel('Cosine Similarity')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].axvline(np.mean(query_retrieval_sim), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(query_retrieval_sim):.3f}')
        axes[1, 2].legend()
        
        # 7. 记忆演化分析（与初始状态比较）
        current_memory_keys = self.model.memory_module.memory_keys.detach().cpu().numpy()
        current_memory_values = self.model.memory_module.memory_values.detach().cpu().numpy()
        
        # 计算记忆槽之间的相似性矩阵
        key_similarity = np.corrcoef(current_memory_keys)
        im3 = axes[2, 0].imshow(key_similarity, cmap='coolwarm', vmin=-1, vmax=1)
        axes[2, 0].set_title('Memory Key Similarity Matrix')
        axes[2, 0].set_xlabel('Memory Slot')
        axes[2, 0].set_ylabel('Memory Slot')
        plt.colorbar(im3, ax=axes[2, 0])
        
        # 8. 记忆值分布
        axes[2, 1].hist(current_memory_values.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[2, 1].set_title('Memory Values Distribution')
        axes[2, 1].set_xlabel('Value')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].axvline(current_memory_values.mean(), color='red', linestyle='--', 
                          label=f'Mean: {current_memory_values.mean():.3f}')
        axes[2, 1].legend()
        
        # 9. 类别特定的记忆使用模式
        class_memory_usage = np.zeros((10, self.config.memory_size))
        for class_id in range(10):
            mask = labels == class_id
            if np.sum(mask) > 0:
                class_memory_usage[class_id] = attention_weights[mask].mean(axis=0)
        
        im4 = axes[2, 2].imshow(class_memory_usage, cmap='viridis', aspect='auto')
        axes[2, 2].set_title('Class-Specific Memory Usage')
        axes[2, 2].set_xlabel('Memory Slot')
        axes[2, 2].set_ylabel('Class')
        axes[2, 2].set_yticks(range(10))
        plt.colorbar(im4, ax=axes[2, 2])
        
        plt.tight_layout()
        
        # 保存结果
        save_path = os.path.join(save_dir, f'memory_analysis_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 M组件可视化已保存: {save_path}")
        
        # 返回统计信息
        stats = {
            'mean_attention_entropy': float(-np.sum(attention_weights * np.log(attention_weights + 1e-8), axis=1).mean()),
            'mixture_diversity': float(-np.sum(gaussian_weights * np.log(gaussian_weights + 1e-8))),
            'query_retrieval_similarity': float(np.mean(query_retrieval_sim))
        }
        
        return stats
    
    def visualize_controller_component(self, data_loader, epoch, save_dir="results"):
        """可视化C组件 - 控制器的融合和决策结果"""
        print(f"🎮 可视化C组件 (Controller) - Epoch {epoch}")
        
        # 收集控制器数据
        controller_outputs = []
        gate_weights = []
        var_codes = []
        memory_codes = []
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for i, (x, y) in enumerate(data_loader):
                if i >= 15:  # 限制批次数量
                    break
                
                x = x.view(x.size(0), -1).to(self.config.device)
                y = y.to(self.config.device)
                
                # 获取完整输出
                output, components = self.model(x)
                
                controller_outputs.append(components['controller']['output'].cpu())
                gate_weights.append(components['controller']['gate_weight'].cpu())
                var_codes.append(components['variational']['code'].cpu())
                memory_codes.append(components['memory']['retrieved'].cpu())
                predictions.append(torch.softmax(output, dim=1).cpu())
                true_labels.append(y.cpu())
        
        # 合并数据
        controller_outputs = torch.cat(controller_outputs, dim=0).numpy()
        gate_weights = torch.cat(gate_weights, dim=0).numpy().squeeze()
        var_codes = torch.cat(var_codes, dim=0).numpy()
        memory_codes = torch.cat(memory_codes, dim=0).numpy()
        predictions = torch.cat(predictions, dim=0).numpy()
        true_labels = torch.cat(true_labels, dim=0).numpy()
        
        # 创建可视化
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle(f'C Component (Controller) - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # 1. 门控权重分布
        axes[0, 0].hist(gate_weights, bins=50, alpha=0.7, color='gold', edgecolor='black')
        axes[0, 0].set_title('Gate Weight Distribution')
        axes[0, 0].set_xlabel('Gate Weight')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(gate_weights.mean(), color='red', linestyle='--', 
                          label=f'Mean: {gate_weights.mean():.3f}')
        axes[0, 0].legend()
        
        # 2. 预测置信度分布
        prediction_confidence = np.max(predictions, axis=1)
        axes[0, 1].hist(prediction_confidence, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 1].set_title('Prediction Confidence Distribution')
        axes[0, 1].set_xlabel('Max Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(prediction_confidence.mean(), color='red', linestyle='--', 
                          label=f'Mean: {prediction_confidence.mean():.3f}')
        axes[0, 1].legend()
        
        # 3. 门控权重vs预测置信度
        scatter = axes[0, 2].scatter(gate_weights, prediction_confidence, 
                                    c=true_labels, cmap='tab10', alpha=0.6)
        axes[0, 2].set_xlabel('Gate Weight')
        axes[0, 2].set_ylabel('Prediction Confidence')
        axes[0, 2].set_title('Gate Weight vs Prediction Confidence')
        plt.colorbar(scatter, ax=axes[0, 2])
        
        # 计算相关系数
        correlation = np.corrcoef(gate_weights, prediction_confidence)[0, 1]
        axes[0, 2].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[0, 2].transAxes, bbox=dict(boxstyle="round", facecolor='white'))
        
        # 4. 分类准确率分析
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = (predicted_labels == true_labels).mean()
        
        # 混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        im1 = axes[1, 0].imshow(cm, cmap='Blues')
        axes[1, 0].set_title(f'Confusion Matrix (Acc: {accuracy:.3f})')
        axes[1, 0].set_xlabel('Predicted Label')
        axes[1, 0].set_ylabel('True Label')
        
        # 添加数值标注
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1, 0].text(j, i, str(cm[i, j]), ha='center', va='center', 
                               color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        # 5. 各类别的门控权重分布
        gate_by_class = []
        for class_id in range(10):
            mask = true_labels == class_id
            if np.sum(mask) > 0:
                class_gates = gate_weights[mask]
                gate_by_class.append(class_gates)
        
        axes[1, 1].boxplot(gate_by_class, labels=range(10))
        axes[1, 1].set_title('Gate Weight Distribution by Class')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Gate Weight')
        
        # 6. 控制器输出的主成分分析
        if controller_outputs.shape[1] > 2:
            pca = PCA(n_components=2)
            controller_pca = pca.fit_transform(controller_outputs)
            explained_var = pca.explained_variance_ratio_
        else:
            controller_pca = controller_outputs
            explained_var = [0.5, 0.5]
        
        scatter2 = axes[1, 2].scatter(controller_pca[:, 0], controller_pca[:, 1], 
                                     c=true_labels, cmap='tab10', alpha=0.6)
        axes[1, 2].set_title(f'Controller Output PCA\n(Explained Var: {explained_var[0]:.2f}, {explained_var[1]:.2f})')
        axes[1, 2].set_xlabel('PC1')
        axes[1, 2].set_ylabel('PC2')
        plt.colorbar(scatter2, ax=axes[1, 2])
        
        # 7. 变分vs记忆编码的贡献分析
        # 计算变分编码和记忆编码对最终输出的相对贡献
        var_norms = np.linalg.norm(var_codes, axis=1)
        memory_norms = np.linalg.norm(memory_codes, axis=1)
        
        axes[2, 0].scatter(var_norms, memory_norms, c=gate_weights, cmap='viridis', alpha=0.6)
        axes[2, 0].set_xlabel('Variational Code Norm')
        axes[2, 0].set_ylabel('Memory Code Norm')
        axes[2, 0].set_title('Code Norms vs Gate Weight')
        
        # 添加对角线
        max_norm = max(var_norms.max(), memory_norms.max())
        axes[2, 0].plot([0, max_norm], [0, max_norm], 'r--', alpha=0.5, label='Equal Contribution')
        axes[2, 0].legend()
        
        # 8. 错误分析
        correct_mask = predicted_labels == true_labels
        incorrect_mask = ~correct_mask
        
        if np.sum(incorrect_mask) > 0:
            axes[2, 1].hist([gate_weights[correct_mask], gate_weights[incorrect_mask]], 
                           bins=30, alpha=0.7, label=['Correct', 'Incorrect'], 
                           color=['green', 'red'], edgecolor='black')
            axes[2, 1].set_title('Gate Weight: Correct vs Incorrect Predictions')
            axes[2, 1].set_xlabel('Gate Weight')
            axes[2, 1].set_ylabel('Frequency')
            axes[2, 1].legend()
        
        # 9. 决策边界可视化（使用t-SNE）
        if controller_outputs.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            controller_tsne = tsne.fit_transform(controller_outputs)
        else:
            controller_tsne = controller_outputs
        
        # 绘制正确和错误预测
        axes[2, 2].scatter(controller_tsne[correct_mask, 0], controller_tsne[correct_mask, 1], 
                          c=true_labels[correct_mask], cmap='tab10', alpha=0.6, 
                          marker='o', s=30, label='Correct')
        axes[2, 2].scatter(controller_tsne[incorrect_mask, 0], controller_tsne[incorrect_mask, 1], 
                          c='red', alpha=0.8, marker='x', s=50, label='Incorrect')
        axes[2, 2].set_title('Decision Space (t-SNE)')
        axes[2, 2].set_xlabel('t-SNE 1')
        axes[2, 2].set_ylabel('t-SNE 2')
        axes[2, 2].legend()
        
        plt.tight_layout()
        
        # 保存结果
        save_path = os.path.join(save_dir, f'controller_analysis_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 C组件可视化已保存: {save_path}")
        
        # 返回统计信息
        stats = {
            'accuracy': float(accuracy),
            'mean_gate_weight': float(gate_weights.mean()),
            'mean_confidence': float(prediction_confidence.mean()),
            'gate_confidence_correlation': float(correlation)
        }
        
        return stats
    
    def visualize_complete_pipeline(self, data_loader, epoch, save_dir="results"):
        """可视化完整的VMC流水线"""
        print(f"🔄 可视化完整VMC流水线 - Epoch {epoch}")
        
        # 获取一个样本批次进行详细分析
        x_sample, y_sample = next(iter(data_loader))
        x_sample = x_sample[:8].view(8, -1).to(self.config.device)  # 只取8个样本
        y_sample = y_sample[:8].to(self.config.device)
        
        with torch.no_grad():
            output, components = self.model(x_sample)
        
        # 创建流水线可视化
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Complete VMC Pipeline - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # 1. 原始输入（重塑为图像）
        for i in range(8):
            row, col = i // 4, i % 4
            img = x_sample[i].cpu().numpy().reshape(28, 28)
            # 反标准化
            img = img * 0.3081 + 0.1307
            img = np.clip(img, 0, 1)
            
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f'Input: {y_sample[i].item()}')
            axes[row, col].axis('off')
            
            # 添加VMC流水线信息
            var_code = components['variational']['code'][i].cpu().numpy()
            memory_attention = components['memory']['attention_weights'][i].cpu().numpy()
            gate_weight = components['controller']['gate_weight'][i].cpu().item()
            pred_probs = torch.softmax(output[i], dim=0).cpu().numpy()
            predicted_class = torch.argmax(output[i]).cpu().item()
            
            # 在图像下方添加文本信息
            info_text = f"Pred: {predicted_class} ({pred_probs[predicted_class]:.2f})\n"
            info_text += f"Gate: {gate_weight:.3f}\n"
            info_text += f"V-norm: {np.linalg.norm(var_code):.2f}\n"
            info_text += f"M-max: {memory_attention.max():.2f}"
            
            axes[row, col].text(0.02, 0.02, info_text, transform=axes[row, col].transAxes,
                               fontsize=8, verticalalignment='bottom',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存结果
        save_path = os.path.join(save_dir, f'complete_pipeline_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 完整流水线可视化已保存: {save_path}")
        
        return save_path

def test_visualizer():
    """测试可视化器"""
    print("🧪 测试VMC可视化器...")
    
    # 这里需要实际的模型和数据来测试
    # 暂时跳过，在main.py中集成测试
    print("可视化器测试需要在完整训练环境中进行")

if __name__ == "__main__":
    test_visualizer()