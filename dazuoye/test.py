import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from model2 import SimpleUNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch.nn.functional as F
import SimpleITK as sitk
import tempfile
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_samples=None):
        self.data_dir = data_dir
        self.transform = transform
        self.max_samples = max_samples
        
        # 检查目录是否存在
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
            
        # 获取所有.mhd文件
        self.mhd_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mhd')])
        if not self.mhd_files:
            raise ValueError(f"在 {data_dir} 中没有找到.mhd文件")
            
        # 如果设置了最大样本数，随机选择部分样本
        if max_samples is not None and max_samples < len(self.mhd_files):
            indices = np.random.choice(len(self.mhd_files), max_samples, replace=False)
            self.mhd_files = [self.mhd_files[i] for i in indices]
            
        # 创建临时目录用于存储转换后的图像
        self.temp_dir = tempfile.mkdtemp()
        print(f"创建临时目录: {self.temp_dir}")
        
    def __len__(self):
        return len(self.mhd_files)
    
    def __getitem__(self, idx):
        try:
            mhd_file = self.mhd_files[idx]
            mhd_path = os.path.join(self.data_dir, mhd_file)
            
            # 读取.mhd文件
            image = sitk.ReadImage(mhd_path)
            image_array = sitk.GetArrayFromImage(image)
            
            # 获取中间切片
            middle_slice = image_array[image_array.shape[0] // 2]
            
            # 归一化到0-255
            middle_slice = ((middle_slice - middle_slice.min()) / 
                          (middle_slice.max() - middle_slice.min()) * 255).astype(np.uint8)
            
            # 转换为RGB
            image_rgb = np.stack([middle_slice] * 3, axis=-1)
            
            # 创建掩码
            mask = np.ones_like(middle_slice, dtype=np.uint8)
            
            # 应用变换
            if self.transform:
                augmented = self.transform(image=image_rgb, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                image = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float()
                mask = torch.from_numpy(mask)
            
            return image, mask
            
        except Exception as e:
            print(f"处理文件 {mhd_file} 时出错: {str(e)}")
            # 返回一个空图像和掩码
            return torch.zeros((3, 512, 512)), torch.zeros((512, 512))
    
    def __del__(self):
        # 清理临时目录
        if hasattr(self, 'temp_dir'):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"清理临时目录时出错: {str(e)}")

def get_test_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def calculate_metrics(pred, target):
    try:
        # 将预测结果转换为二值图像
        pred = (pred > 0.5).float()
        
        # 计算Dice系数
        intersection = (pred * target).sum()
        dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
        
        # 计算IoU
        union = pred.sum() + target.sum() - intersection
        iou = intersection / (union + 1e-8)
        
        return dice.item(), iou.item()
    except Exception as e:
        print(f"计算指标时出错: {str(e)}")
        return 0.0, 0.0

def visualize_results(images, pred_masks, true_masks, save_dir, num_samples=5):
    """
    可视化分割结果
    :param images: 原始图像 [B, C, H, W]
    :param pred_masks: 预测掩码 [B, 1, H, W]
    :param true_masks: 真实掩码 [B, H, W]
    :param save_dir: 保存目录
    :param num_samples: 要可视化的样本数量
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # 选择要可视化的样本
        indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
        
        for idx in indices:
            # 获取图像和掩码
            image = images[idx].cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
            pred_mask = pred_masks[idx].cpu().numpy().squeeze()  # [H, W]
            true_mask = true_masks[idx].cpu().numpy()  # [H, W]
            
            # 反归一化图像
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)
            
            # 创建可视化图像
            plt.figure(figsize=(15, 5))
            
            # 显示原始图像
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title('原始图像')
            plt.axis('off')
            
            # 显示预测掩码
            plt.subplot(1, 3, 2)
            plt.imshow(pred_mask, cmap='gray')
            plt.title('预测掩码')
            plt.axis('off')
            
            # 显示真实掩码
            plt.subplot(1, 3, 3)
            plt.imshow(true_mask, cmap='gray')
            plt.title('真实掩码')
            plt.axis('off')
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'segmentation_result_{idx}.png'))
            plt.close()
            
    except Exception as e:
        print(f"可视化结果时出错: {str(e)}")

def evaluate_model(model, test_loader, device):
    model.eval()
    total_dice = 0
    total_iou = 0
    num_samples = 0
    
    # 用于存储每个样本的指标
    dice_scores = []
    iou_scores = []
    
    # 用于存储可视化结果
    all_images = []
    all_pred_masks = []
    all_true_masks = []
    
    try:
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="评估进度"):
                try:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    # 前向传播
                    outputs = model(images)
                    probs = torch.sigmoid(outputs)
                    pred_masks = (probs > 0.5).float()
                    
                    # 存储结果用于可视化
                    all_images.extend(images.cpu())
                    all_pred_masks.extend(pred_masks.cpu())
                    all_true_masks.extend(masks.cpu())
                    
                    # 计算每个样本的指标
                    for pred, target in zip(pred_masks, masks):
                        dice, iou = calculate_metrics(pred, target)
                        dice_scores.append(dice)
                        iou_scores.append(iou)
                        total_dice += dice
                        total_iou += iou
                        num_samples += 1
                    
                    # 清理GPU内存
                    del images, masks, outputs, probs, pred_masks
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"处理批次时出错: {str(e)}")
                    continue
                
                # 定期进行垃圾回收
                if num_samples % 100 == 0:
                    gc.collect()
    
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
    
    # 计算平均指标
    avg_dice = total_dice / max(num_samples, 1)
    avg_iou = total_iou / max(num_samples, 1)
    
    return avg_dice, avg_iou, dice_scores, iou_scores, all_images, all_pred_masks, all_true_masks

def plot_metrics(dice_scores, iou_scores, save_dir):
    try:
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建图表
        plt.figure(figsize=(15, 5))
        
        # 绘制Dice系数分布
        plt.subplot(1, 2, 1)
        plt.hist(dice_scores, bins=50, alpha=0.75)
        plt.title('Dice系数分布')
        plt.xlabel('Dice系数')
        plt.ylabel('样本数量')
        plt.grid(True, alpha=0.3)
        
        # 绘制IoU分布
        plt.subplot(1, 2, 2)
        plt.hist(iou_scores, bins=50, alpha=0.75)
        plt.title('IoU分布')
        plt.xlabel('IoU')
        plt.ylabel('样本数量')
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_distribution.png'))
        plt.close()
        
        # 创建箱线图
        plt.figure(figsize=(10, 5))
        plt.boxplot([dice_scores, iou_scores], labels=['Dice系数', 'IoU'])
        plt.title('评估指标箱线图')
        plt.ylabel('得分')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'metrics_boxplot.png'))
        plt.close()
        
    except Exception as e:
        print(f"绘制图表时出错: {str(e)}")

def main():
    try:
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 设置路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "shuju", "seg-lungs-LUNA16", "seg-lungs-LUNA16")
        model_path = os.path.join(current_dir, "best_model.pth")
        results_dir = os.path.join(current_dir, "results")
        visualization_dir = os.path.join(results_dir, "visualization")
        
        print(f"数据目录: {data_dir}")
        print(f"模型文件路径: {model_path}")
        
        # 创建测试数据集和数据加载器，限制最大样本数为100
        test_dataset = TestDataset(
            data_dir=data_dir,
            transform=get_test_transforms(),
            max_samples=100  # 限制最大样本数
        )
        print(f"成功加载测试数据集，共 {len(test_dataset)} 个样本")
        
        # 根据设备类型设置pin_memory
        pin_memory = device.type == 'cuda'
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            pin_memory=pin_memory
        )
        
        # 创建模型
        model = SimpleUNet(in_channels=3, out_channels=1).to(device)
        print("使用随机初始化的模型")
        
        # 评估模型
        print("开始评估模型...")
        avg_dice, avg_iou, dice_scores, iou_scores, all_images, all_pred_masks, all_true_masks = evaluate_model(model, test_loader, device)
        
        # 打印结果
        print("\n评估结果:")
        print(f"平均Dice系数: {avg_dice:.4f}")
        print(f"平均IoU: {avg_iou:.4f}")
        
        # 绘制并保存图表
        print("\n正在生成评估指标图表...")
        plot_metrics(dice_scores, iou_scores, results_dir)
        print(f"图表已保存到: {results_dir}")
        
        # 可视化分割结果
        print("\n正在生成分割结果可视化...")
        visualize_results(all_images, all_pred_masks, all_true_masks, visualization_dir)
        print(f"可视化结果已保存到: {visualization_dir}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 清理资源
        if 'test_dataset' in locals():
            del test_dataset
        if 'test_loader' in locals():
            del test_loader
        if 'model' in locals():
            del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 