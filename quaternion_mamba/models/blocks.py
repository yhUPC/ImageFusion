"""
Building Blocks for Quaternion Mamba Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quaternion.ops import QuaternionTensor
from ..quaternion.layers import (
    QuaternionLinear, 
    QuaternionConv2d, 
    QuaternionLayerNorm,
    QuaternionBatchNorm2d
)
from ..quaternion.activations import QuaternionGELU
from .qssm import QuaternionSSMBlock


class InputProjection(nn.Module):
    """
    将 IR + RGB 投影到四元数空间
    """
    def __init__(self, output_dim: int = 64):
        super().__init__()
        
        # IR 通道投影 (1 -> output_dim)
        self.ir_proj = nn.Conv2d(1, output_dim, kernel_size=3, padding=1)
        
        # RGB 通道投影 (3 -> 3*output_dim)
        self.rgb_proj = nn.Conv2d(3, output_dim * 3, kernel_size=3, padding=1)
        
    def forward(self, ir: torch.Tensor, rgb: torch.Tensor) -> QuaternionTensor:
        """
        Args:
            ir: [B, 1, H, W] 红外图像
            rgb: [B, 3, H, W] 可见光图像
        
        Returns:
            QuaternionTensor [B, C, H, W]
        """
        # 投影 IR 到实部
        r = self.ir_proj(ir)  # [B, C, H, W]
        
        # 投影 RGB 到虚部
        rgb_proj = self.rgb_proj(rgb)  # [B, 3C, H, W]
        i, j, k = torch.chunk(rgb_proj, 3, dim=1)  # 各 [B, C, H, W]
        
        return QuaternionTensor(r, i, j, k)


class QuaternionFFN(nn.Module):
    """
    四元数前馈网络
    """
    def __init__(self, d_model: int, expand: int = 4, dropout: float = 0.0):
        super().__init__()
        
        d_ff = d_model * expand
        
        self.fc1 = QuaternionLinear(d_model, d_ff)
        self.activation = QuaternionGELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = QuaternionLinear(d_ff, d_model)
    
    def forward(self, q_input: QuaternionTensor) -> QuaternionTensor:
        x = self.fc1(q_input)
        x = self.activation(x)
        x_r = self.dropout(x.r)
        x_i = self.dropout(x.i)
        x_j = self.dropout(x.j)
        x_k = self.dropout(x.k)
        x = QuaternionTensor(x_r, x_i, x_j, x_k)
        x = self.fc2(x)
        return x


class QMambaBlock(nn.Module):
    """
    完整的 Q-Mamba Block
    包含: Q-SSM + FFN + Residual
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        ffn_expand: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Q-SSM
        self.norm1 = QuaternionLayerNorm(d_model)
        self.qssm = QuaternionSSMBlock(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            dropout=dropout,
        )
        
        # FFN
        self.norm2 = QuaternionLayerNorm(d_model)
        self.ffn = QuaternionFFN(
            d_model=d_model,
            expand=ffn_expand,
            dropout=dropout,
        )
    
    def forward(self, q_input: QuaternionTensor) -> QuaternionTensor:
        """
        前向传播
        """
        # Q-SSM with residual
        x = q_input + self.qssm(self.norm1(q_input))
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x


class QuaternionDownsample(nn.Module):
    """
    四元数下采样层
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv = QuaternionConv2d(
            in_channels, out_channels,
            kernel_size=3, stride=2, padding=1
        )
        self.norm = QuaternionBatchNorm2d(out_channels)
        self.activation = QuaternionGELU()
    
    def forward(self, q_input: QuaternionTensor) -> QuaternionTensor:
        x = self.conv(q_input)
        x = self.norm(x)
        x = self.activation(x)
        return x


class QuaternionUpsample(nn.Module):
    """
    四元数上采样层
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv = QuaternionConv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.norm = QuaternionBatchNorm2d(out_channels)
        self.activation = QuaternionGELU()
    
    def forward(self, q_input: QuaternionTensor) -> QuaternionTensor:
        # 双线性上采样
        B, C, H, W = q_input.r.shape
        r_up = F.interpolate(q_input.r, scale_factor=2, mode='bilinear', align_corners=False)
        i_up = F.interpolate(q_input.i, scale_factor=2, mode='bilinear', align_corners=False)
        j_up = F.interpolate(q_input.j, scale_factor=2, mode='bilinear', align_corners=False)
        k_up = F.interpolate(q_input.k, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = QuaternionTensor(r_up, i_up, j_up, k_up)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class QuaternionFusionModule(nn.Module):
    """
    四元数融合模块
    利用四元数的内在几何结构进行模态融合
    """
    def __init__(self, channels: int):
        super().__init__()
        
        # 注意力机制：学习模态权重
        self.attention = nn.Sequential(
            QuaternionConv2d(channels, channels // 4, kernel_size=1),
            QuaternionGELU(),
            QuaternionConv2d(channels // 4, channels, kernel_size=1),
        )
        
    def forward(self, q_features: QuaternionTensor) -> QuaternionTensor:
        """
        四元数融合
        利用 Hamilton 乘积的旋转特性
        """
        # 计算注意力权重
        attn = self.attention(q_features)
        
        # 通过四元数乘法进行融合
        # 这会自动考虑 IR 和 RGB 之间的几何关系
        fused = q_features.hamilton_product(attn)
        
        # 归一化
        fused = fused.normalize()
        
        return fused