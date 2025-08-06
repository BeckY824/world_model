"""
主入口文件 - VAE项目的统一入口
"""
import argparse
import os
import sys
import torch
from datetime import datetime

from config import Config
from train import VAETrainer, quick_train
from visualize import VAEVisualizer, load_and_visualize
from vae import create_vae_model
from dataset import create_dataloader

def print_banner():
    """打印项目横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                        VAE CelebA 项目                        ║
    ║                 变分自编码器人脸生成模型                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def train_mode():
    """训练模式"""
    print("🚀 启动训练模式")
    Config.print_config()
    
    try:
        trained_model, trainer = quick_train()
        
        if trained_model is not None:
            print("\n✅ 训练成功完成！")
            
            # 询问是否进行可视化
            if input("\n是否进行结果可视化？(y/n): ").lower() == 'y':
                print("🎨 开始可视化...")
                visualizer = VAEVisualizer(trained_model, trainer.dataloader)
                visualizer.comprehensive_analysis()
                
                # 绘制训练曲线
                visualizer.plot_training_curves(
                    trainer.train_losses,
                    trainer.recon_losses,
                    trainer.kl_losses
                )
        else:
            print("❌ 训练失败！")
            
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")

def visualize_mode(model_path=None):
    """可视化模式"""
    print("🎨 启动可视化模式")
    
    if model_path is None:
        model_path = "vae_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行训练模式: python main.py --mode train")
        return
    
    try:
        load_and_visualize(model_path)
        print("✅ 可视化完成！")
        
    except Exception as e:
        print(f"❌ 可视化过程中出错: {e}")

def test_mode():
    """测试模式 - 测试各个模块"""
    print("🧪 启动测试模式")
    
    try:
        print("\n1. 测试配置...")
        Config.print_config()
        
        print("\n2. 测试数据加载...")
        dataloader = create_dataloader()
        if dataloader:
            batch = next(iter(dataloader))
            print(f"   ✅ 数据加载成功，批次形状: {batch[0].shape}")
        else:
            print("   ❌ 数据加载失败")
            return
        
        print("\n3. 测试模型创建...")
        model = create_vae_model()
        test_input = batch[0][:2].to(Config.device)
        output = model(test_input)
        print(f"   ✅ 模型测试成功，输出形状: {[x.shape for x in output]}")
        
        print("\n4. 测试训练器...")
        trainer = VAETrainer(model, dataloader)
        print("   ✅ 训练器创建成功")
        
        print("\n5. 测试可视化器...")
        visualizer = VAEVisualizer(model, dataloader)
        print("   ✅ 可视化器创建成功")
        
        print("\n🎉 所有模块测试通过！")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")

def interactive_mode():
    """交互模式"""
    print("🎮 启动交互模式")
    
    while True:
        print("\n" + "="*50)
        print("请选择操作:")
        print("1. 开始训练")
        print("2. 可视化结果")
        print("3. 生成样本")
        print("4. 测试模块")
        print("5. 查看配置")
        print("0. 退出")
        print("="*50)
        
        choice = input("请输入选择 (0-5): ").strip()
        
        if choice == '1':
            train_mode()
        elif choice == '2':
            model_path = input("模型文件路径 (回车使用默认): ").strip()
            if not model_path:
                model_path = "vae_model.pth"
            visualize_mode(model_path)
        elif choice == '3':
            generate_samples_interactive()
        elif choice == '4':
            test_mode()
        elif choice == '5':
            Config.print_config()
        elif choice == '0':
            print("👋 再见！")
            break
        else:
            print("❌ 无效选择，请重新输入")

def generate_samples_interactive():
    """交互式生成样本"""
    model_path = "vae_model.pth"
    
    if not os.path.exists(model_path):
        print("❌ 未找到训练好的模型，请先进行训练")
        return
    
    try:
        # 加载模型
        model = create_vae_model()
        checkpoint = torch.load(model_path, map_location=Config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        visualizer = VAEVisualizer(model)
        
        while True:
            print("\n生成样本选项:")
            print("1. 生成随机样本")
            print("2. 潜在空间插值")
            print("3. 重建对比")
            print("0. 返回主菜单")
            
            choice = input("请选择: ").strip()
            
            if choice == '1':
                num_samples = int(input("生成样本数量 (默认16): ") or "16")
                visualizer.generate_samples(num_samples)
            elif choice == '2':
                num_steps = int(input("插值步数 (默认10): ") or "10")
                visualizer.latent_space_interpolation(num_steps)
            elif choice == '3':
                num_pairs = int(input("对比对数 (默认8): ") or "8")
                visualizer.show_reconstruction(num_pairs)
            elif choice == '0':
                break
            else:
                print("❌ 无效选择")
                
    except Exception as e:
        print(f"❌ 生成样本时出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VAE CelebA 项目")
    parser.add_argument(
        '--mode', 
        choices=['train', 'visualize', 'test', 'interactive'],
        default='interactive',
        help='运行模式'
    )
    parser.add_argument(
        '--model', 
        type=str,
        default='vae_model.pth',
        help='模型文件路径'
    )
    parser.add_argument(
        '--config',
        action='store_true',
        help='显示配置信息'
    )
    
    args = parser.parse_args()
    
    # 打印横幅
    print_banner()
    
    # 显示配置
    if args.config:
        Config.print_config()
        return
    
    # 根据模式运行
    if args.mode == 'train':
        train_mode()
    elif args.mode == 'visualize':
        visualize_mode(args.model)
    elif args.mode == 'test':
        test_mode()
    elif args.mode == 'interactive':
        interactive_mode()

if __name__ == "__main__":
    main()