"""
Quaternion Neural Network Layers
四元数神经网络层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import QuaternionTensor, quaternion_init


class QuaternionLinear(nn.Module):
    """
    四元数全连接层
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 四元数权重：4个实值矩阵
        self.W_r = nn.Parameter(torch.empty(out_features, in_features))
        self.W_i = nn.Parameter(torch.empty(out_features, in_features))
        self.W_j = nn.Parameter(torch.empty(out_features, in_features))
        self.W_k = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features))
            self.bias_i = nn.Parameter(torch.zeros(out_features))
            self.bias_j = nn.Parameter(torch.zeros(out_features))
            self.bias_k = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_r', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        # 使用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.W_r, a=0, mode='fan_in')
        nn.init.kaiming_uniform_(self.W_i, a=0, mode='fan_in')
        nn.init.kaiming_uniform_(self.W_j, a=0, mode='fan_in')
        nn.init.kaiming_uniform_(self.W_k, a=0, mode='fan_in')
    
    def forward(self, q_input: QuaternionTensor) -> QuaternionTensor:
        """
        前向传播
        q_input: [B, ..., in_features]
        output: [B, ..., out_features]
        """
        # 构造权重四元数
        W = QuaternionTensor(self.W_r, self.W_i, self.W_j, self.W_k)
        
        # 四元数矩阵乘法的分量形式
        # (W × q) 的实部 = W_r*q_r - W_i*q_i - W_j*q_j - W_k*q_k
        r_out = (F.linear(q_input.r, self.W_r) - 
                 F.linear(q_input.i, self.W_i) - 
                 F.linear(q_input.j, self.W_j) - 
                 F.linear(q_input.k, self.W_k))
        
        i_out = (F.linear(q_input.r, self.W_i) + 
                 F.linear(q_input.i, self.W_r) + 
                 F.linear(q_input.j, self.W_k) - 
                 F.linear(q_input.k, self.W_j))
        
        j_out = (F.linear(q_input.r, self.W_j) - 
                 F.linear(q_input.i, self.W_k) + 
                 F.linear(q_input.j, self.W_r) + 
                 F.linear(q_input.k, self.W_i))
        
        k_out = (F.linear(q_input.r, self.W_k) + 
                 F.linear(q_input.i, self.W_j) - 
                 F.linear(q_input.j, self.W_i) + 
                 F.linear(q_input.k, self.W_r))
        
        # 添加偏置
        if self.bias_r is not None:
            r_out = r_out + self.bias_r
            i_out = i_out + self.bias_i
            j_out = j_out + self.bias_j
            k_out = k_out + self.bias_k
        
        return QuaternionTensor(r_out, i_out, j_out, k_out)


class QuaternionConv2d(nn.Module):
    """
    四元数卷积层
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, stride: int = 1, 
                 padding: int = 0, bias: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 四元数卷积核
        self.W_r = nn.Parameter(torch.empty(out_channels, in_channels, 
                                           kernel_size, kernel_size))
        self.W_i = nn.Parameter(torch.empty(out_channels, in_channels, 
                                           kernel_size, kernel_size))
        self.W_j = nn.Parameter(torch.empty(out_channels, in_channels, 
                                           kernel_size, kernel_size))
        self.W_k = nn.Parameter(torch.empty(out_channels, in_channels, 
                                           kernel_size, kernel_size))
        
        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_channels))
            self.bias_i = nn.Parameter(torch.zeros(out_channels))
            self.bias_j = nn.Parameter(torch.zeros(out_channels))
            self.bias_k = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias_r', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_r, a=0, mode='fan_in')
        nn.init.kaiming_uniform_(self.W_i, a=0, mode='fan_in')
        nn.init.kaiming_uniform_(self.W_j, a=0, mode='fan_in')
        nn.init.kaiming_uniform_(self.W_k, a=0, mode='fan_in')
    
    def forward(self, q_input: QuaternionTensor) -> QuaternionTensor:
        """
        四元数卷积
        q_input: [B, C_in, H, W]
        output: [B, C_out, H', W']
        """
        # Hamilton 乘积的卷积形式
        r_out = (F.conv2d(q_input.r, self.W_r, stride=self.stride, padding=self.padding) -
                 F.conv2d(q_input.i, self.W_i, stride=self.stride, padding=self.padding) -
                 F.conv2d(q_input.j, self.W_j, stride=self.stride, padding=self.padding) -
                 F.conv2d(q_input.k, self.W_k, stride=self.stride, padding=self.padding))
        
        i_out = (F.conv2d(q_input.r, self.W_i, stride=self.stride, padding=self.padding) +
                 F.conv2d(q_input.i, self.W_r, stride=self.stride, padding=self.padding) +
                 F.conv2d(q_input.j, self.W_k, stride=self.stride, padding=self.padding) -
                 F.conv2d(q_input.k, self.W_j, stride=self.stride, padding=self.padding))
        
        j_out = (F.conv2d(q_input.r, self.W_j, stride=self.stride, padding=self.padding) -
                 F.conv2d(q_input.i, self.W_k, stride=self.stride, padding=self.padding) +
                 F.conv2d(q_input.j, self.W_r, stride=self.stride, padding=self.padding) +
                 F.conv2d(q_input.k, self.W_i, stride=self.stride, padding=self.padding))
        
        k_out = (F.conv2d(q_input.r, self.W_k, stride=self.stride, padding=self.padding) +
                 F.conv2d(q_input.i, self.W_j, stride=self.stride, padding=self.padding) -
                 F.conv2d(q_input.j, self.W_i, stride=self.stride, padding=self.padding) +
                 F.conv2d(q_input.k, self.W_r, stride=self.stride, padding=self.padding))
        
        if self.bias_r is not None:
            r_out = r_out + self.bias_r.view(1, -1, 1, 1)
            i_out = i_out + self.bias_i.view(1, -1, 1, 1)
            j_out = j_out + self.bias_j.view(1, -1, 1, 1)
            k_out = k_out + self.bias_k.view(1, -1, 1, 1)
        
        return QuaternionTensor(r_out, i_out, j_out, k_out)


class QuaternionLayerNorm(nn.Module):
    """
    四元数层归一化
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # 可学习参数
        self.gamma_r = nn.Parameter(torch.ones(normalized_shape))
        self.gamma_i = nn.Parameter(torch.ones(normalized_shape))
        self.gamma_j = nn.Parameter(torch.ones(normalized_shape))
        self.gamma_k = nn.Parameter(torch.ones(normalized_shape))
        
        self.beta_r = nn.Parameter(torch.zeros(normalized_shape))
        self.beta_i = nn.Parameter(torch.zeros(normalized_shape))
        self.beta_j = nn.Parameter(torch.zeros(normalized_shape))
        self.beta_k = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, q_input: QuaternionTensor) -> QuaternionTensor:
        """
        四元数归一化
        对四元数的模进行归一化
        """
        # 计算四元数的模
        norm = q_input.norm()  # [B, C, H, W] or [B, L, D]
        
        # 归一化
        mean = norm.mean(dim=-1, keepdim=True)
        var = norm.var(dim=-1, keepdim=True)
        norm_normalized = (norm - mean) / torch.sqrt(var + self.eps)
        
        # 保持方向，调整模长
        direction = q_input.normalize()
        
        # 应用可学习参数
        r_out = direction.r * norm_normalized * self.gamma_r + self.beta_r
        i_out = direction.i * norm_normalized * self.gamma_i + self.beta_i
        j_out = direction.j * norm_normalized * self.gamma_j + self.beta_j
        k_out = direction.k * norm_normalized * self.gamma_k + self.beta_k
        
        return QuaternionTensor(r_out, i_out, j_out, k_out)


class QuaternionBatchNorm2d(nn.Module):
    """
    四元数批归一化
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta_r = nn.Parameter(torch.zeros(num_features))
        self.beta_i = nn.Parameter(torch.zeros(num_features))
        self.beta_j = nn.Parameter(torch.zeros(num_features))
        self.beta_k = nn.Parameter(torch.zeros(num_features))
        
        # 运行统计
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, q_input: QuaternionTensor) -> QuaternionTensor:
        """
        四元数批归一化
        """
        if self.training:
            # 计算批统计
            norm = q_input.norm()  # [B, C, H, W]
            mean = norm.mean(dim=[0, 2, 3])
            var = norm.var(dim=[0, 2, 3])
            
            # 更新运行统计
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        norm = q_input.norm()
        norm_normalized = (norm - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        
        # 保持方向
        direction = q_input.normalize()
        
        # 缩放和平移
        gamma = self.gamma.view(1, -1, 1, 1)
        beta_r = self.beta_r.view(1, -1, 1, 1)
        beta_i = self.beta_i.view(1, -1, 1, 1)
        beta_j = self.beta_j.view(1, -1, 1, 1)
        beta_k = self.beta_k.view(1, -1, 1, 1)
        
        r_out = direction.r * norm_normalized * gamma + beta_r
        i_out = direction.i * norm_normalized * gamma + beta_i
        j_out = direction.j * norm_normalized * gamma + beta_j
        k_out = direction.k * norm_normalized * gamma + beta_k
        
        return QuaternionTensor(r_out, i_out, j_out, k_out)