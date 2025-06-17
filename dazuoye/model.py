import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        # 1x1 卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 卷积，dilation=6
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 3x3 卷积，dilation=12
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # 3x3 卷积，dilation=18
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        # 全局平均池化
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 输出卷积
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.relu(self.bn_out(self.conv_out(x)))
        
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(256, num_classes, 1)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepLabV3Plus, self).__init__()
        
        # 使用预训练的ResNet50作为backbone
        backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # ASPP模块
        self.aspp = ASPP(2048, 256)
        
        # Decoder模块
        self.decoder = Decoder(256, num_classes)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # 获取backbone特征
        x = self.backbone(x)
        
        # ASPP处理
        x = self.aspp(x)
        
        # Decoder处理
        x = self.decoder(x)
        
        # 上采样到原始大小
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        
        return x 