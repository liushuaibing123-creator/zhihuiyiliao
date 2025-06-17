import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from model import DeepLabV3Plus
from data_processing import prepare_data

def calculate_accuracy(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    # 计算每个通道的平均准确率
    correct = (pred == target).float().mean(dim=(2, 3))
    return correct.mean().item()

def dice_coefficient(pred, target):
    smooth = 1e-5
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    # 计算每个通道的Dice系数
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return torch.clamp(dice.mean(), 0, 1)

def iou_score(pred, target):
    smooth = 1e-5
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    # 计算每个通道的IoU
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    return (intersection + smooth) / (union + smooth)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        # 计算每个通道的Dice损失
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth)
        return torch.clamp(1 - dice.mean(), 0, 1)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_dice = 0
        train_acc = 0
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            masks = masks.to(device)
            
            # 调整mask的维度顺序从[B, H, W, C]到[B, C, H, W]
            masks = masks.permute(0, 3, 1, 2)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_coefficient(outputs, masks).item()
            train_acc += calculate_accuracy(outputs, masks)
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_acc /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_dice = 0
        val_acc = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                # 调整mask的维度顺序
                masks = masks.permute(0, 3, 1, 2)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, masks).item()
                val_acc += calculate_accuracy(outputs, masks)
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_acc /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train Dice: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val Dice: {val_dice:.4f}')
        print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    # 绘制损失曲线
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def evaluate_model(model, test_loader, device):
    model.eval()
    dice_scores = []
    iou_scores = []
    accuracies = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            masks = masks.to(device)
            
            # 调整mask的维度顺序
            masks = masks.permute(0, 3, 1, 2)
            
            outputs = model(images)
            dice = dice_coefficient(outputs, masks).item()
            iou = iou_score(outputs, masks).item()
            acc = calculate_accuracy(outputs, masks)
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            accuracies.append(acc)
    
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    mean_acc = np.mean(accuracies)
    
    print(f'Test Results:')
    print(f'Mean Accuracy: {mean_acc:.4f}')
    print(f'Mean Dice Coefficient: {mean_dice:.4f}')
    print(f'Mean IoU Score: {mean_iou:.4f}')

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 准备数据
    data_dir = 'shuju'  # 数据目录
    train_loader, val_loader, test_loader = prepare_data(
        data_dir, 
        batch_size=16,  # 减小batch size以适应CPU
        num_workers=0,  # CPU模式下使用单进程
        pin_memory=False  # CPU模式下禁用pin_memory
    )
    
    # 创建模型
    model = DeepLabV3Plus(num_classes=3).to(device)
    
    # 定义损失函数和优化器
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    # 训练模型
    num_epochs = 10
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)
    
    # 加载最佳模型进行评估
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()