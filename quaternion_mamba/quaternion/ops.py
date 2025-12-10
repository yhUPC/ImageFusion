"""
Quaternion Operations
四元数基础运算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class QuaternionTensor:
    """
    四元数张量封装
    q = r + xi + yj + zk
    """
    def __init__(self, r: torch.Tensor, i: torch.Tensor, 
                 j: torch.Tensor, k: torch.Tensor):
        """
        Args:
            r, i, j, k: 四元数的四个分量，形状必须相同
        """
        assert r.shape == i.shape == j.shape == k.shape, \
            f"All quaternion components must have the same shape, got r:{r.shape}, i:{i.shape}, j:{j.shape}, k:{k.shape}"
        
        self.r = r  # Real part (实部)
        self.i = i  # Imaginary i (虚部i)
        self.j = j  # Imaginary j (虚部j)
        self.k = k  # Imaginary k (虚部k)
        self.shape = r.shape
        self.device = r.device
        self.dtype = r.dtype
    
    def hamilton_product(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
        """
        Hamilton 四元数乘法
        
        公式:
        (a + bi + cj + dk) × (e + fi + gj + hk) = 
            (ae - bf - cg - dh) +
            (af + be + ch - dg)i +
            (ag - bh + ce + df)j +
            (ah + bg - cf + de)k
        """
        r = self.r * other.r - self.i * other.i - self.j * other.j - self.k * other.k
        i = self.r * other.i + self.i * other.r + self.j * other.k - self.k * other.j
        j = self.r * other.j - self.i * other.k + self.j * other.r + self.k * other.i
        k = self.r * other.k + self.i * other.j - self.j * other.i + self.k * other.r
        
        return QuaternionTensor(r, i, j, k)
    
    def conjugate(self) -> 'QuaternionTensor':
        """四元数共轭: q* = r - xi - yj - zk"""
        return QuaternionTensor(self.r, -self.i, -self.j, -self.k)
    
    def norm(self) -> torch.Tensor:
        """
        四元数范数: ||q|| = sqrt(r² + i² + j² + k²)
        返回形状与输入相同
        """
        return torch.sqrt(self.r**2 + self.i**2 + self.j**2 + self.k**2 + 1e-8)
    
    def normalize(self) -> 'QuaternionTensor':
        """
        四元数归一化
        """
        n = self.norm()  # 形状: [B, C, H, W] 或 [B, L, D]
        
        # 确保 n 的形状与分量相同，用于广播
        # 不需要 unsqueeze，因为 norm() 已经返回正确的形状
        return QuaternionTensor(
            self.r / n,
            self.i / n,
            self.j / n,
            self.k / n
        )
    
    def __add__(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
        """四元数加法"""
        return QuaternionTensor(
            self.r + other.r,
            self.i + other.i,
            self.j + other.j,
            self.k + other.k
        )
    
    def __sub__(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
        """四元数减法"""
        return QuaternionTensor(
            self.r - other.r,
            self.i - other.i,
            self.j - other.j,
            self.k - other.k
        )
    
    def __mul__(self, scalar) -> 'QuaternionTensor':
        """标量乘法或四元数乘法"""
        if isinstance(scalar, (int, float)):
            return QuaternionTensor(
                self.r * scalar,
                self.i * scalar,
                self.j * scalar,
                self.k * scalar
            )
        elif isinstance(scalar, torch.Tensor) and scalar.numel() == 1:
            return QuaternionTensor(
                self.r * scalar,
                self.i * scalar,
                self.j * scalar,
                self.k * scalar
            )
        elif isinstance(scalar, QuaternionTensor):
            return self.hamilton_product(scalar)
        else:
            raise TypeError(f"Unsupported type for multiplication: {type(scalar)}")
    
    def __rmul__(self, scalar) -> 'QuaternionTensor':
        """右乘"""
        return self.__mul__(scalar)
    
    def to_real_tensor(self) -> torch.Tensor:
        """
        转换为实数张量 [B, 4C, H, W] 或 [B, L, 4D]
        用于与标准PyTorch操作兼容
        """
        return torch.cat([self.r, self.i, self.j, self.k], dim=-3 if len(self.shape) == 4 else -1)
    
    @staticmethod
    def from_real_tensor(x: torch.Tensor) -> 'QuaternionTensor':
        """
        从实数张量构造四元数
        """
        if len(x.shape) == 4:
            # [B, 4C, H, W] -> 4 × [B, C, H, W]
            B, C4, H, W = x.shape
            assert C4 % 4 == 0, f"Channel dimension must be divisible by 4, got {C4}"
            C = C4 // 4
            r, i, j, k = torch.split(x, C, dim=1)
        elif len(x.shape) == 3:
            # [B, L, 4D] -> 4 × [B, L, D]
            B, L, D4 = x.shape
            assert D4 % 4 == 0, f"Feature dimension must be divisible by 4, got {D4}"
            D = D4 // 4
            r, i, j, k = torch.split(x, D, dim=2)
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")
        
        return QuaternionTensor(r, i, j, k)
    
    @staticmethod
    def zeros(*shape, device='cpu', dtype=torch.float32) -> 'QuaternionTensor':
        """创建零四元数"""
        r = torch.zeros(*shape, device=device, dtype=dtype)
        i = torch.zeros(*shape, device=device, dtype=dtype)
        j = torch.zeros(*shape, device=device, dtype=dtype)
        k = torch.zeros(*shape, device=device, dtype=dtype)
        return QuaternionTensor(r, i, j, k)
    
    @staticmethod
    def randn(*shape, device='cpu', dtype=torch.float32) -> 'QuaternionTensor':
        """创建随机四元数"""
        r = torch.randn(*shape, device=device, dtype=dtype)
        i = torch.randn(*shape, device=device, dtype=dtype)
        j = torch.randn(*shape, device=device, dtype=dtype)
        k = torch.randn(*shape, device=device, dtype=dtype)
        return QuaternionTensor(r, i, j, k)
    
    @staticmethod
    def stack(tensors, dim=0) -> 'QuaternionTensor':
        """堆叠多个四元数张量"""
        r_list = [t.r for t in tensors]
        i_list = [t.i for t in tensors]
        j_list = [t.j for t in tensors]
        k_list = [t.k for t in tensors]
        
        return QuaternionTensor(
            torch.stack(r_list, dim=dim),
            torch.stack(i_list, dim=dim),
            torch.stack(j_list, dim=dim),
            torch.stack(k_list, dim=dim)
        )
    
    def get_timestep(self, t: int) -> 'QuaternionTensor':
        """获取序列中的第 t 个时间步"""
        return QuaternionTensor(
            self.r[:, t],
            self.i[:, t],
            self.j[:, t],
            self.k[:, t]
        )
    
    def split_channels(self, n: int):
        """
        将通道维度分成 n 份
        返回 n 个 QuaternionTensor
        """
        if len(self.shape) == 4:
            # [B, C, H, W]
            r_splits = torch.chunk(self.r, n, dim=1)
            i_splits = torch.chunk(self.i, n, dim=1)
            j_splits = torch.chunk(self.j, n, dim=1)
            k_splits = torch.chunk(self.k, n, dim=1)
        elif len(self.shape) == 3:
            # [B, L, D]
            r_splits = torch.chunk(self.r, n, dim=2)
            i_splits = torch.chunk(self.i, n, dim=2)
            j_splits = torch.chunk(self.j, n, dim=2)
            k_splits = torch.chunk(self.k, n, dim=2)
        else:
            raise ValueError(f"Unsupported shape: {self.shape}")
        
        return [QuaternionTensor(r, i, j, k) 
                for r, i, j, k in zip(r_splits, i_splits, j_splits, k_splits)]
    
    def detach(self) -> 'QuaternionTensor':
        """Detach from computation graph"""
        return QuaternionTensor(
            self.r.detach(),
            self.i.detach(),
            self.j.detach(),
            self.k.detach()
        )
    
    def clone(self) -> 'QuaternionTensor':
        """Clone quaternion tensor"""
        return QuaternionTensor(
            self.r.clone(),
            self.i.clone(),
            self.j.clone(),
            self.k.clone()
        )
    
    def to(self, device) -> 'QuaternionTensor':
        """Move to device"""
        return QuaternionTensor(
            self.r.to(device),
            self.i.to(device),
            self.j.to(device),
            self.k.to(device)
        )
    
    def __repr__(self):
        return f"QuaternionTensor(shape={self.shape}, device={self.device}, dtype={self.dtype})"


def quaternion_init(shape: Tuple, gain: float = 1.0, 
                   device='cpu', dtype=torch.float32) -> QuaternionTensor:
    """
    四元数权重初始化
    使用标准正交初始化保证四元数的范数
    """
    # 生成四个独立的正态分布
    r = torch.randn(*shape, device=device, dtype=dtype)
    i = torch.randn(*shape, device=device, dtype=dtype)
    j = torch.randn(*shape, device=device, dtype=dtype)
    k = torch.randn(*shape, device=device, dtype=dtype)
    
    # 归一化并缩放
    q = QuaternionTensor(r, i, j, k)
    q = q.normalize() * gain
    
    return q