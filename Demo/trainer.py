"""
VMC训练模块 - 整合V、M、C三个组件的训练流程
"""
import torch
import torch.optim as optim
import numpy as np
import os
from datetime import datetime

from config import VMCConfig
from vmc_model import create_vmc_model
from dataset import VMCDataset
from visualizer import VMCVisualizer

class VMCTrainer:
    """VMC训练器"""
    
    def __init__(self, config=None):
        self.config = config if config else VMCConfig()
        
        # 创建数据集
        self.dataset_manager = VMCDataset()
        self.train_loader, self.test_loader = self.dataset_manager.create_dataloaders()
        
        # 创建模型
        self.model = create_vmc_model(self.config)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs, eta_min=1e-6
        )
        
        # 创建可视化器
        self.visualizer = VMCVisualizer(self.model, self.config)
        
        # 训练统计
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.component_stats = {
            'variational': [],
            'memory': [],
            'controller': []
        }
        
        print(f"🚀 VMC训练器初始化完成")
        print(f"   模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   训练数据: {len(self.train_loader.dataset)}")
        print(f"   测试数据: {len(self.test_loader.dataset)}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_classification_loss = 0
        total_kl_loss = 0
        total_memory_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            # 数据预处理
            x = x.view(x.size(0), -1).to(self.config.device)
            y = y.to(self.config.device)
            
            # 前向传播
            output, components = self.model(x)
            
            # 计算损失
            losses = self.model.compute_loss(x, y, output, components)
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += losses['total_loss'].item()
            total_classification_loss += losses['classification_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
            total_memory_loss += losses['memory_loss'].item()
            
            # 计算准确率
            predicted = torch.argmax(output, dim=1)
            correct_predictions += (predicted == y).sum().item()
            total_samples += y.size(0)
            
            # 定期更新记忆
            if batch_idx % 10 == 0:
                memory_query = self.model.var_to_memory(components['variational']['code'])
                self.model.memory_module.update_memory(
                    memory_query, components['memory']['retrieved'], learning_rate=0.01
                )
            
            # 打印训练进度
            if batch_idx % 100 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f'   Epoch {epoch+1}/{self.config.epochs}, '
                      f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {losses["total_loss"].item():.4f}, '
                      f'Acc: {100.*correct_predictions/total_samples:.2f}%, '
                      f'LR: {current_lr:.2e}')
        
        # 计算平均值
        avg_loss = total_loss / len(self.train_loader)
        avg_classification_loss = total_classification_loss / len(self.train_loader)
        avg_kl_loss = total_kl_loss / len(self.train_loader)
        avg_memory_loss = total_memory_loss / len(self.train_loader)
        train_accuracy = 100. * correct_predictions / total_samples
        
        return {
            'avg_loss': avg_loss,
            'classification_loss': avg_classification_loss,
            'kl_loss': avg_kl_loss,
            'memory_loss': avg_memory_loss,
            'train_accuracy': train_accuracy
        }
    
    def evaluate(self, epoch):
        """评估模型性能"""
        self.model.eval()
        
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.view(x.size(0), -1).to(self.config.device)
                y = y.to(self.config.device)
                
                # 前向传播
                output, components = self.model(x)
                
                # 计算损失
                losses = self.model.compute_loss(x, y, output, components)
                total_loss += losses['total_loss'].item()
                
                # 计算准确率
                predicted = torch.argmax(output, dim=1)
                correct_predictions += (predicted == y).sum().item()
                total_samples += y.size(0)
        
        test_accuracy = 100. * correct_predictions / total_samples
        avg_test_loss = total_loss / len(self.test_loader)
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': avg_test_loss
        }
    
    def visualize_components(self, epoch, save_dir="results", skip_visualization=False):
        """可视化各个组件"""
        if skip_visualization:
            print(f"🎨 跳过组件可视化 - Epoch {epoch} (演示模式)")
            return
            
        print(f"🎨 开始可视化各组件 - Epoch {epoch}")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # 可视化V组件
            print("   📊 可视化V组件...")
            v_stats = self.visualizer.visualize_variational_component(
                self.test_loader, epoch, save_dir
            )
            self.component_stats['variational'].append(v_stats)
            
            # 可视化M组件  
            print("   🧠 可视化M组件...")
            m_stats = self.visualizer.visualize_memory_component(
                self.test_loader, epoch, save_dir
            )
            self.component_stats['memory'].append(m_stats)
            
            # 可视化C组件
            print("   🎮 可视化C组件...")
            c_stats = self.visualizer.visualize_controller_component(
                self.test_loader, epoch, save_dir
            )
            self.component_stats['controller'].append(c_stats)
            
            # 可视化完整流水线
            print("   🔄 可视化完整流水线...")
            pipeline_path = self.visualizer.visualize_complete_pipeline(
                self.test_loader, epoch, save_dir
            )
            
            print(f"✅ 所有组件可视化完成 - Epoch {epoch}")
            
            return {
                'variational_stats': v_stats,
                'memory_stats': m_stats,
                'controller_stats': c_stats,
                'pipeline_path': pipeline_path
            }
        except Exception as e:
            print(f"   ❌ 可视化过程中出错: {e}")
            return {
                'variational_stats': None,
                'memory_stats': None,
                'controller_stats': None,
                'pipeline_path': None
            }
    
    def save_checkpoint(self, epoch, save_dir="checkpoints"):
        """保存训练检查点"""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'component_stats': self.component_stats,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(save_dir, f'vmc_checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        print(f"💾 检查点已保存: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """加载训练检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.test_accuracies = checkpoint.get('test_accuracies', [])
        self.component_stats = checkpoint.get('component_stats', {
            'variational': [], 'memory': [], 'controller': []
        })
        
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"📂 检查点已加载: {checkpoint_path}")
        print(f"   继续从epoch {start_epoch}开始训练")
        
        return start_epoch
    
    def train(self, start_epoch=0, demo_mode=False):
        """完整训练流程"""
        print(f"🎯 开始VMC训练")
        print(f"   训练轮数: {self.config.epochs}")
        print(f"   开始epoch: {start_epoch}")
        print("=" * 80)
        
        start_time = datetime.now()
        
        try:
            for epoch in range(start_epoch, self.config.epochs):
                print(f"\n📈 Epoch {epoch+1}/{self.config.epochs}")
                print("-" * 60)
                
                # 训练一个epoch
                train_stats = self.train_epoch(epoch)
                
                # 评估模型
                eval_stats = self.evaluate(epoch)
                
                # 更新学习率
                self.scheduler.step()
                
                # 记录统计数据
                self.train_losses.append(train_stats['avg_loss'])
                self.train_accuracies.append(train_stats['train_accuracy'])
                self.test_accuracies.append(eval_stats['test_accuracy'])
                
                # 打印epoch总结
                print(f"\n✅ Epoch {epoch+1} 完成:")
                print(f"   训练损失: {train_stats['avg_loss']:.4f}")
                print(f"   分类损失: {train_stats['classification_loss']:.4f}")
                print(f"   KL损失: {train_stats['kl_loss']:.4f}")
                print(f"   记忆损失: {train_stats['memory_loss']:.4f}")
                print(f"   训练准确率: {train_stats['train_accuracy']:.2f}%")
                print(f"   测试准确率: {eval_stats['test_accuracy']:.2f}%")
                
                # 定期可视化组件
                if (epoch + 1) % self.config.save_interval == 0:
                    vis_stats = self.visualize_components(epoch + 1, skip_visualization=demo_mode)
                    
                    # 打印组件统计
                    if vis_stats and not demo_mode:
                        print(f"\n🔍 组件分析 - Epoch {epoch+1}:")
                        if vis_stats['variational_stats']:
                            print(f"   V组件 - 平均KL散度: {vis_stats['variational_stats']['mean_kl_div']:.4f}")
                        if vis_stats['memory_stats']:
                            print(f"   M组件 - 注意力熵: {vis_stats['memory_stats']['mean_attention_entropy']:.4f}")
                        if vis_stats['controller_stats']:
                            print(f"   C组件 - 分类准确率: {vis_stats['controller_stats']['accuracy']:.4f}")
                
                # 定期保存检查点
                if (epoch + 1) % (self.config.save_interval * 2) == 0:
                    self.save_checkpoint(epoch + 1)
                
                print("-" * 60)
        
        except KeyboardInterrupt:
            print("\n⚠️ 训练被用户中断")
            self.save_checkpoint(epoch + 1)
        
        except Exception as e:
            print(f"\n❌ 训练过程中出错: {e}")
            self.save_checkpoint(epoch + 1)
            raise
        
        finally:
            end_time = datetime.now()
            training_time = end_time - start_time
            
            print(f"\n🎉 训练完成!")
            print(f"   总训练时间: {training_time}")
            print(f"   最终训练准确率: {self.train_accuracies[-1]:.2f}%")
            print(f"   最终测试准确率: {self.test_accuracies[-1]:.2f}%")
            
            # 最终可视化
            if not demo_mode:
                print(f"\n🎨 生成最终可视化结果...")
                final_vis = self.visualize_components(self.config.epochs)
            else:
                print(f"\n🎨 跳过内置可视化 (演示模式)")
                final_vis = None
            
            # 保存最终模型
            final_checkpoint = self.save_checkpoint(self.config.epochs)
            
            return {
                'model': self.model,
                'training_time': training_time,
                'final_train_accuracy': self.train_accuracies[-1],
                'final_test_accuracy': self.test_accuracies[-1],
                'checkpoint_path': final_checkpoint,
                'visualization_stats': final_vis
            }

def quick_train():
    """快速训练函数"""
    print("🚀 启动VMC快速训练...")
    
    # 显示配置
    VMCConfig.print_config()
    
    try:
        # 创建训练器
        trainer = VMCTrainer()
        
        # 开始训练
        results = trainer.train()
        
        print("✅ 快速训练成功完成!")
        return results
        
    except Exception as e:
        print(f"❌ 快速训练失败: {e}")
        return None

if __name__ == "__main__":
    # 直接运行训练
    results = quick_train()