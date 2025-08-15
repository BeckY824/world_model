"""
数据集模块 - 处理CelebA数据集加载
"""
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import Config

class CustomCelebADataset(Dataset):
    """自定义CelebA数据集类"""
    
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        
        Args:
            root_dir (str): 图像文件根目录
            transform: 图像预处理变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg"))
        self.image_paths.sort()  # 确保顺序一致
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # 返回图像和虚拟标签（VAE不需要标签）

def get_transform():
    """获取数据预处理变换"""
    return transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
    ])

def create_dataloader():
    """
    创建CelebA数据加载器
    
    Returns:
        DataLoader: 数据加载器，如果失败返回None
    """
    try:
        # 检查数据目录
        img_dir = Config.data_root
        if not os.path.exists(img_dir):
            # 兼容性：如果文件在data根目录下
            img_dir = "./data/img_align_celeba"
            
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"图像目录不存在: {img_dir}")
            
        # 创建数据集和数据加载器
        transform = get_transform()
        dataset = CustomCelebADataset(img_dir, transform=transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=Config.batch_size, 
            shuffle=True, 
            num_workers=0,  # macOS兼容性
            pin_memory=False  # MPS不支持pin_memory
        )
        
        print(f"✅ 成功加载CelebA数据集，共有 {len(dataset)} 张图像")
        print(f"📂 图像目录: {img_dir}")
        print(f"📊 批次大小: {Config.batch_size}")
        print(f"📊 批次数量: {len(dataloader)}")
        
        return dataloader
        
    except FileNotFoundError as e:
        print(f"❌ 未找到CelebA图像文件: {e}")
        print("请确保图像文件在以下位置之一:")
        print("  ./data/celeba/img_align_celeba/")
        print("  ./data/img_align_celeba/")
        return None
        
    except Exception as e:
        print(f"❌ 加载CelebA数据集时出错: {e}")
        return None

if __name__ == "__main__":
    # 测试数据集加载
    print("测试数据集加载...")
    dataloader = create_dataloader()
    if dataloader:
        # 测试第一个batch
        for batch_idx, (images, labels) in enumerate(dataloader):
            print(f"第一个batch - 图像形状: {images.shape}, 标签形状: {labels.shape}")
            break
        print("✅ 数据集测试通过！")