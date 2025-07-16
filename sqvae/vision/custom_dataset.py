import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import glob

class MicroDopplerDataset(Dataset):
    def __init__(self, root, transform=None, split='train', train_ratio=0.7, val_ratio=0.15):
        """
        微多普勒数据集加载器
        
        参数:
            root (str): 数据集根目录
            transform (callable, optional): 图像变换
            split (str): 'train', 'val', 或 'test'
            train_ratio (float): 训练集比例
            val_ratio (float): 验证集比例
        """
        self.root = root
        self.transform = transform
        self.split = split
        
        # 查找所有用户文件夹 (ID_1, ID_2, ..., ID_31)
        user_folders = []
        for item in os.listdir(root):
            item_path = os.path.join(root, item)
            if os.path.isdir(item_path) and item.startswith('ID_'):
                user_folders.append(item)

        user_folders = sorted(user_folders)  # ID_1, ID_2, ..., ID_31
        self.id_to_label = {folder: i for i, folder in enumerate(user_folders)}

        # 按用户划分数据集 (避免数据泄露)
        np.random.seed(42)  # 固定随机种子
        user_indices = np.random.permutation(len(user_folders))
        train_user_end = int(train_ratio * len(user_folders))
        val_user_end = int((train_ratio + val_ratio) * len(user_folders))

        if split == 'train':
            selected_users = [user_folders[i] for i in user_indices[:train_user_end]]
        elif split == 'val':
            selected_users = [user_folders[i] for i in user_indices[train_user_end:val_user_end]]
        else:  # test
            selected_users = [user_folders[i] for i in user_indices[val_user_end:]]

        # 收集选定用户的所有图像
        self.image_paths = []
        for user in selected_users:
            user_path = os.path.join(root, user)
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                self.image_paths.extend(glob.glob(os.path.join(user_path, ext)))

        print(f"{split.upper()} 集: {len(selected_users)} 用户, {len(self.image_paths)} 图像")
        print(f"用户列表: {selected_users}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
            
        if self.transform:
            image = self.transform(image)
        
        # 提取用户ID作为标签
        user_id = os.path.basename(os.path.dirname(image_path))
        label = self.id_to_label[user_id]
        
        return image, label
