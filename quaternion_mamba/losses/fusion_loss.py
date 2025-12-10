"""
Loss functions for image fusion
图像融合的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional


class SSIMLoss(nn.Module):
    """
    结构相似性损失
    """
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        """
        Args:
            img1, img2: [B, C, H, W]
        Returns:
            SSIM loss (1 - SSIM)
        """
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        return 1 - self.ssim(img1, img2, window, self.window_size, channel, self.size_average)


class GradientLoss(nn.Module):
    """
    梯度损失 - 保持纹理细节
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        计算梯度差异
        """
        # Sobel 算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        # 扩展到所有通道
        B, C, H, W = pred.shape
        sobel_x = sobel_x.repeat(C, 1, 1, 1)
        sobel_y = sobel_y.repeat(C, 1, 1, 1)
        
        # 计算梯度
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=C)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=C)
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        
        target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=C)
        target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=C)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        return F.l1_loss(pred_grad, target_grad)


class IntensityLoss(nn.Module):
    """
    强度损失 - 保持红外的热辐射信息
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, fused, ir):
        """
        Args:
            fused: [B, 3, H, W] 融合图像
            ir: [B, 1, H, W] 红外图像
        """
        # 将 RGB 转换为灰度
        fused_gray = 0.299 * fused[:, 0] + 0.587 * fused[:, 1] + 0.114 * fused[:, 2]
        fused_gray = fused_gray.unsqueeze(1)
        
        return F.l1_loss(fused_gray, ir)


class VGGPerceptualLoss(nn.Module):
    """
    VGG 感知损失
    """
    def __init__(self, resize: bool = True):
        super().__init__()
        
        # 使用预训练的 VGG16
        vgg = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:23]).eval()
        
        # 冻结参数
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.resize = resize
        
        # 归一化参数 (ImageNet)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred, target):
        """
        Args:
            pred, target: [B, 3, H, W]
        """
        # 归一化
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # Resize 到 VGG 输入尺寸
        if self.resize:
            pred = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 提取特征
        pred_features = self.features(pred)
        target_features = self.features(target)
        
        return F.mse_loss(pred_features, target_features)


class FusionLoss(nn.Module):
    """
    完整的融合损失
    
    L_total = λ1*L_ssim + λ2*L_gradient + λ3*L_intensity + λ4*L_perceptual
    """
    def __init__(
        self,
        ssim_weight: float = 1.0,
        gradient_weight: float = 10.0,
        intensity_weight: float = 5.0,
        perceptual_weight: float = 0.1,
    ):
        super().__init__()
        
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
        self.intensity_weight = intensity_weight
        self.perceptual_weight = perceptual_weight
        
        self.ssim_loss = SSIMLoss()
        self.gradient_loss = GradientLoss()
        self.intensity_loss = IntensityLoss()
        self.perceptual_loss = VGGPerceptualLoss()
    
    def forward(self, fused, ir, rgb):
        """
        Args:
            fused: [B, 3, H, W] 融合图像
            ir: [B, 1, H, W] 红外图像
            rgb: [B, 3, H, W] 可见光图像
        
        Returns:
            loss: 总损失
            loss_dict: 各个损失分量
        """
        # 1. SSIM 损失 (与两个输入的结构相似性)
        loss_ssim_ir = self.ssim_loss(fused, ir.repeat(1, 3, 1, 1))
        loss_ssim_rgb = self.ssim_loss(fused, rgb)
        loss_ssim = (loss_ssim_ir + loss_ssim_rgb) / 2
        
        # 2. 梯度损失 (保持 RGB 的纹理)
        loss_grad = self.gradient_loss(fused, rgb)
        
        # 3. 强度损失 (保持 IR 的热辐射)
        loss_intensity = self.intensity_loss(fused, ir)
        
        # 4. 感知损失
        loss_percep = self.perceptual_loss(fused, rgb)
        
        # 总损失
        loss_total = (
            self.ssim_weight * loss_ssim +
            self.gradient_weight * loss_grad +
            self.intensity_weight * loss_intensity +
            self.perceptual_weight * loss_percep
        )
        
        loss_dict = {
            'loss_total': loss_total.item(),
            'loss_ssim': loss_ssim.item(),
            'loss_gradient': loss_grad.item(),
            'loss_intensity': loss_intensity.item(),
            'loss_perceptual': loss_percep.item(),
        }
        
        return loss_total, loss_dict