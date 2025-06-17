import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LungDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        image_path = self.image_paths[idx]
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)
        
        # 选择中间切片
        middle_slice = image.shape[0] // 2
        image = image[middle_slice]
        
        # 归一化到0-1并转换为float32
        image = (image - image.min()) / (image.max() - image.min())
        image = image.astype(np.float32)
        
        # 转换为三通道
        image = np.stack([image] * 3, axis=-1)
        
        # 如果没有mask，使用图像本身作为mask（自监督学习）
        if self.mask_paths is None:
            mask = image.copy()
        else:
            mask_path = self.mask_paths[idx]
            mask = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(mask)
            mask = mask[middle_slice]
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = mask.astype(np.float32)
            mask = np.stack([mask] * 3, axis=-1)
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask

def prepare_data(data_dir, batch_size=32, num_workers=4, pin_memory=None):
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if pin_memory is None:
        pin_memory = device.type == 'cuda'
    
    # 获取所有.mhd文件
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mhd'):
                image_paths.append(os.path.join(root, file))
    
    print(f"找到 {len(image_paths)} 个.mhd文件")
    
    # 分割数据集
    train_paths, temp_paths = train_test_split(image_paths, test_size=0.3, random_state=42)
    val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)
    
    # 定义数据增强
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.0625, 0.0625),
            rotate=(-45, 45),
            p=0.5
        ),
        A.OneOf([
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                p=0.5
            ),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(
                distort_limit=1,
                p=0.5
            ),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),  # 使用默认参数
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # 创建数据集
    train_dataset = LungDataset(train_paths, transform=train_transform)
    val_dataset = LungDataset(val_paths, transform=val_transform)
    test_dataset = LungDataset(test_paths, transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader