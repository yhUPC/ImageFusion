"""
Quaternion Selective State Space Model (Q-SSM)
四元数选择性状态空间模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from typing import Optional

from ..quaternion.ops import QuaternionTensor
from ..quaternion.layers import QuaternionLinear


class QuaternionSSM(nn.Module):
    """
    四元数选择性状态空间模型
    
    公式:
    h(t) = A ⊗ h(t-1) ⊕ B ⊗ x(t)
    y(t) = C ⊗ h(t)
    
    其中 ⊗ 是四元数乘法, ⊕ 是四元数加法
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        if dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank
        
        # 输入投影 (四元数)
        self.in_proj = QuaternionLinear(self.d_model, self.d_inner * 2, bias=bias)
        
        # 卷积层 (用于局部依赖)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner * 4,  # 四元数 = 4个实数通道
            out_channels=self.d_inner * 4,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner * 4,
            padding=d_conv - 1,
        )
        
        # x_proj: 从输入生成 Δ, B, C
        # 注意：这里我们用实数层，因为 Δ, B, C 是控制参数
        self.x_proj = nn.Linear(
            self.d_inner * 4,  # 四元数展平后
            self.dt_rank + self.d_state * 2,
            bias=False,
        )
        
        # dt_proj: 将 dt_rank 投影到 d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner * 4, bias=True)
        
        # A 参数: 状态转移矩阵
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner * 4,
        )
        self.A_log = nn.Parameter(torch.log(A))
        
        # D 参数: skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner * 4))
        
        # 输出投影
        self.out_proj = QuaternionLinear(self.d_inner, self.d_model, bias=bias)
        
        # 初始化 dt_proj
        self.dt_proj.weight.data = torch.randn_like(self.dt_proj.weight) * dt_scale
        self.dt_proj.bias.data = torch.rand(self.d_inner * 4) * (dt_max - dt_min) + dt_min
    
    def forward(self, q_input: QuaternionTensor) -> QuaternionTensor:
        """
        前向传播
        
        Args:
            q_input: QuaternionTensor [B, L, D]
        
        Returns:
            QuaternionTensor [B, L, D]
        """
        B, L, D = q_input.r.shape
        
        # 1. 输入投影并分离为 x 和 z (用于门控)
        xz = self.in_proj(q_input)  # [B, L, 2*d_inner]
        
        # 转换为实数张量进行分离
        xz_real = xz.to_real_tensor()  # [B, L, 2*d_inner*4]
        x, z = xz_real.chunk(2, dim=-1)  # 各 [B, L, d_inner*4]
        
        # 2. 1D 卷积 (在序列长度维度)
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L]  # 移除padding
        x = rearrange(x, 'b d l -> b l d')
        
        # 3. 激活
        x = F.silu(x)
        
        # 4. SSM 操作
        y = self.selective_scan(x)
        
        # 5. 门控
        z = F.silu(z)
        y = y * z
        
        # 6. 转回四元数并输出投影
        y_quat = QuaternionTensor.from_real_tensor(
            rearrange(y, 'b l d -> b l d')
        )
        output = self.out_proj(y_quat)
        
        return output
    
    def selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        选择性扫描算法
        
        Args:
            x: [B, L, d_inner*4] 实数张量
        
        Returns:
            [B, L, d_inner*4]
        """
        B, L, D = x.shape
        
        # 1. 生成 Δ, B, C
        x_dbl = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]
        delta, B_param, C_param = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        
        # 2. 计算 Δ (时间步长)
        delta = F.softplus(self.dt_proj(delta))  # [B, L, D]
        
        # 3. 获取 A
        A = -torch.exp(self.A_log.float())  # [D, d_state]
        
        # 4. 离散化: A_bar = exp(Δ * A)
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))  # [B, L, D, d_state]
        deltaB = torch.einsum('bld,bln->bldn', delta, B_param)  # [B, L, D, d_state]
        
        # 5. 选择性扫描 (串行实现，可以用并行scan优化)
        # 修复：正确获取 B 的 int 值
        batch_size = x.shape[0]  # 这是 int
        
        h = torch.zeros(batch_size, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for i in range(L):
            # h = A_bar * h + B_bar * x
            h = deltaA[:, i] * h + deltaB[:, i] * x[:, i].unsqueeze(-1)
            
            # y = C * h
            y = torch.einsum('bdn,bn->bd', h, C_param[:, i])
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # [B, L, D]
        
        # 6. 添加 skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


class QuaternionSSMBlock(nn.Module):
    """
    完整的 Q-SSM Block，包含归一化和残差连接
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        from ..quaternion.layers import QuaternionLayerNorm
        
        self.norm = QuaternionLayerNorm(d_model)
        self.qssm = QuaternionSSM(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q_input: QuaternionTensor) -> QuaternionTensor:
        """
        前向传播，包含残差连接
        """
        # Pre-normalization
        x_norm = self.norm(q_input)
        
        # Q-SSM
        x_ssm = self.qssm(x_norm)
        
        # Dropout + Residual
        x_ssm_r = self.dropout(x_ssm.r)
        x_ssm_i = self.dropout(x_ssm.i)
        x_ssm_j = self.dropout(x_ssm.j)
        x_ssm_k = self.dropout(x_ssm.k)
        
        output = QuaternionTensor(
            q_input.r + x_ssm_r,
            q_input.i + x_ssm_i,
            q_input.j + x_ssm_j,
            q_input.k + x_ssm_k,
        )
        
        return output