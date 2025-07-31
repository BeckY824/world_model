#!/usr/bin/env python3
"""
检查CelebA数据集是否正确放置的脚本
"""
import os
from pathlib import Path

def check_celeba_dataset():
    """检查CelebA数据集文件是否存在"""
    print("检查CelebA数据集...")
    
    data_root = Path("./data")
    celeba_root = data_root / "celeba"
    
    # 需要检查的文件
    required_files = [
        "list_attr_celeba.txt",
        "list_bbox_celeba.txt", 
        "list_landmarks_align_celeba.txt",
        "list_eval_partition.txt"
    ]
    
    # 图像文件夹
    img_folder = celeba_root / "img_align_celeba"
    
    print(f"数据集根目录: {celeba_root.absolute()}")
    print(f"图像文件夹: {img_folder.absolute()}")
    
    # 检查根目录
    if not celeba_root.exists():
        print("❌ CelebA根目录不存在")
        print(f"请创建目录: {celeba_root.absolute()}")
        return False
    
    # 检查元数据文件
    missing_files = []
    for file in required_files:
        file_path = celeba_root / file
        if file_path.exists():
            print(f"✅ 找到文件: {file}")
        else:
            print(f"❌ 缺失文件: {file}")
            missing_files.append(file)
    
    # 检查图像文件夹
    if img_folder.exists():
        # 统计jpg文件数量
        jpg_files = list(img_folder.glob("*.jpg"))
        print(f"✅ 找到图像文件夹: img_align_celeba")
        print(f"📊 图像文件数量: {len(jpg_files)} 张")
        
        if len(jpg_files) == 0:
            print("⚠️  图像文件夹为空")
        elif len(jpg_files) < 200000:  # CelebA大约有202,599张图像
            print("⚠️  图像文件数量可能不完整")
        else:
            print("✅ 图像文件数量正常")
    else:
        print("❌ 缺失图像文件夹: img_align_celeba")
        missing_files.append("img_align_celeba/")
    
    if missing_files:
        print(f"\n❌ 数据集不完整，缺失: {', '.join(missing_files)}")
        print("\n正确的目录结构应该是:")
        print("./data/celeba/")
        print("├── img_align_celeba/")
        print("│   ├── 000001.jpg")
        print("│   ├── 000002.jpg")
        print("│   └── ... (约20万张图片)")
        print("├── list_attr_celeba.txt")
        print("├── list_bbox_celeba.txt")
        print("├── list_landmarks_align_celeba.txt")
        print("└── list_eval_partition.txt")
        print("\n请确保所有文件都已正确解压到对应位置。")
        return False
    else:
        print("\n✅ CelebA数据集检查通过！可以开始训练了。")
        return True

if __name__ == "__main__":
    check_celeba_dataset()