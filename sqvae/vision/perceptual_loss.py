import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    """
    感知损失模块
    使用预训练的VGG网络提取特征，计算感知损失
    """
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], weights=[1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        
        # 加载预训练的VGG16
        vgg = models.vgg16(pretrained=True).features
        
        # 冻结VGG参数
        for param in vgg.parameters():
            param.requires_grad = False
            
        # 定义特征提取层
        self.layer_name_mapping = {
            'relu1_2': 3,   # conv1_2 -> relu1_2
            'relu2_2': 8,   # conv2_2 -> relu2_2  
            'relu3_3': 15,  # conv3_3 -> relu3_3
            'relu4_3': 22,  # conv4_3 -> relu4_3
        }
        
        self.layers = layers
        self.weights = weights
        
        # 构建特征提取器
        self.feature_extractors = nn.ModuleDict()
        for layer in layers:
            layer_idx = self.layer_name_mapping[layer]
            self.feature_extractors[layer] = nn.Sequential(*list(vgg.children())[:layer_idx+1])
    
    def forward(self, pred, target):
        """
        计算感知损失
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
        
        Returns:
            perceptual_loss: 感知损失值
        """
        # 确保输入是3通道
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)
            
        # VGG需要ImageNet标准化
        pred = self.normalize_imagenet(pred)
        target = self.normalize_imagenet(target)
        
        total_loss = 0.0
        
        for i, layer in enumerate(self.layers):
            # 提取特征
            pred_features = self.feature_extractors[layer](pred)
            target_features = self.feature_extractors[layer](target)
            
            # 计算L2损失
            layer_loss = F.mse_loss(pred_features, target_features)
            total_loss += self.weights[i] * layer_loss
            
        return total_loss
    
    def normalize_imagenet(self, x):
        """
        ImageNet标准化
        """
        # ImageNet均值和标准差
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        # 假设输入已经在[0,1]范围内
        x = (x - mean) / std
        return x


class CombinedLoss(nn.Module):
    """
    组合损失：MSE + 感知损失
    """
    def __init__(self, mse_weight=1.0, perceptual_weight=0.1, perceptual_layers=['relu2_2', 'relu3_3']):
        super(CombinedLoss, self).__init__()
        
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        
        # 初始化感知损失
        self.perceptual_loss = PerceptualLoss(layers=perceptual_layers)
        
    def forward(self, pred, target):
        """
        计算组合损失
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
        
        Returns:
            total_loss: 总损失
            mse_loss: MSE损失
            perc_loss: 感知损失
        """
        # MSE损失
        mse_loss = F.mse_loss(pred, target)
        
        # 感知损失
        perc_loss = self.perceptual_loss(pred, target)
        
        # 组合损失
        total_loss = self.mse_weight * mse_loss + self.perceptual_weight * perc_loss
        
        return total_loss, mse_loss, perc_loss


class MicroDopplerPerceptualLoss(nn.Module):
    """
    专门为微多普勒时频图设计的感知损失
    使用较浅的网络层，更适合时频图特征
    """
    def __init__(self, perceptual_weight=0.05):
        super(MicroDopplerPerceptualLoss, self).__init__()
        
        self.perceptual_weight = perceptual_weight
        
        # 使用较浅的VGG层，更适合时频图
        self.perceptual_loss = PerceptualLoss(
            layers=['relu1_2', 'relu2_2'], 
            weights=[1.0, 0.5]
        )
        
    def forward(self, pred, target):
        """
        微多普勒专用组合损失
        """
        # MSE损失
        mse_loss = F.mse_loss(pred, target)
        
        # 感知损失（权重较小，避免过度影响）
        perc_loss = self.perceptual_loss(pred, target)
        
        # 组合损失
        total_loss = mse_loss + self.perceptual_weight * perc_loss
        
        return total_loss, mse_loss, perc_loss
