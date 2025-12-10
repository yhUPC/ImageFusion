"""
Quaternion Activation Functions
四元数激活函数
"""

import torch
import torch.nn as nn
from .ops import QuaternionTensor


class QuaternionReLU(nn.Module):
    """
    四元数 ReLU
    对四元数的每个分量分别应用 ReLU
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, q: QuaternionTensor) -> QuaternionTensor:
        return QuaternionTensor(
            torch.relu(q.r),
            torch.relu(q.i),
            torch.relu(q.j),
            torch.relu(q.k)
        )


class QuaternionGELU(nn.Module):
    """
    四元数 GELU
    """
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()
    
    def forward(self, q: QuaternionTensor) -> QuaternionTensor:
        return QuaternionTensor(
            self.gelu(q.r),
            self.gelu(q.i),
            self.gelu(q.j),
            self.gelu(q.k)
        )


class QuaternionSigmoid(nn.Module):
    """
    四元数 Sigmoid
    使用分裂激活: 保持四元数的单位性
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, q: QuaternionTensor) -> QuaternionTensor:
        # 对模长应用 sigmoid
        norm = q.norm()
        norm_activated = torch.sigmoid(norm)
        
        # 保持方向不变
        direction = q.normalize()
        
        return QuaternionTensor(
            direction.r * norm_activated,
            direction.i * norm_activated,
            direction.j * norm_activated,
            direction.k * norm_activated
        )


class QuaternionTanh(nn.Module):
    """
    四元数 Tanh
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, q: QuaternionTensor) -> QuaternionTensor:
        return QuaternionTensor(
            torch.tanh(q.r),
            torch.tanh(q.i),
            torch.tanh(q.j),
            torch.tanh(q.k)
        )