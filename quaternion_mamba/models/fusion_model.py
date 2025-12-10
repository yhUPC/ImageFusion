"""
Complete Quaternion Mamba Fusion Model
完整的四元数 Mamba 融合模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..quaternion.ops import QuaternionTensor
from ..quaternion.layers import QuaternionConv2d
from .blocks import (
    InputProjection,
    QMambaBlock,
    QuaternionDownsample,
    QuaternionUpsample,
    QuaternionFusionModule,
)


class QuaternionMambaFusion(nn.Module):
    """
    Quaternion Mamba for Multi-Modal Image Fusion
    
    架构:
    1. Input Projection: IR + RGB -> Quaternion
    2. Encoder: 多层 Q-Mamba blocks with downsampling
    3. Fusion: Quaternion fusion module
    4. Decoder: 上采样 + Q-Mamba blocks
    5. Output: Quaternion -> RGB image
    """
    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 4,  # 1 (IR) + 3 (RGB)
        out_channels: int = 3,  # RGB output
        embed_dim: int = 64,
        depths: list = [2, 2, 4, 2],
        d_state: int = 64,
        expand: int = 2,
        ffn_expand: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        
        # 1. 输入投影层
        self.input_proj = InputProjection(output_dim=embed_dim)
        
        # 2. 编码器
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        dim = embed_dim
        for i, depth in enumerate(depths[:-1]):
            # Q-Mamba blocks
            blocks = nn.ModuleList([
                QMambaBlock(
                    d_model=dim,
                    d_state=d_state,
                    expand=expand,
                    ffn_expand=ffn_expand,
                    dropout=dropout,
                )
                for _ in range(depth)
            ])
            self.encoder_layers.append(blocks)
            
            # Downsample
            self.downsample_layers.append(
                QuaternionDownsample(dim, dim * 2)
            )
            dim = dim * 2
        
        # 3. Bottleneck (最深层)
        self.bottleneck = nn.ModuleList([
            QMambaBlock(
                d_model=dim,
                d_state=d_state,
                expand=expand,
                ffn_expand=ffn_expand,
                dropout=dropout,
            )
            for _ in range(depths[-1])
        ])
        
        # 4. 融合模块
        self.fusion = QuaternionFusionModule(channels=dim)
        
        # 5. 解码器
        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        for i in range(self.num_layers - 1, 0, -1):
            # Upsample
            self.upsample_layers.append(
                QuaternionUpsample(dim, dim // 2)
            )
            dim = dim // 2
            
            # Q-Mamba blocks
            blocks = nn.ModuleList([
                QMambaBlock(
                    d_model=dim,
                    d_state=d_state,
                    expand=expand,
                    ffn_expand=ffn_expand,
                    dropout=dropout,
                )
                for _ in range(depths[i-1])
            ])
            self.decoder_layers.append(blocks)
        
        # 6. 输出投影
        self.output_proj = nn.Sequential(
            QuaternionConv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            QuaternionConv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
        )
        
        # 最终输出层（从四元数到RGB）
        self.final_conv = nn.Conv2d(
            (embed_dim // 2) * 4,  # 四元数展平
            out_channels,
            kernel_size=1
        )
    
    def forward(self, ir: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            ir: [B, 1, H, W] 红外图像
            rgb: [B, 3, H, W] 可见光图像
        
        Returns:
            fused: [B, 3, H, W] 融合后的 RGB 图像
        """
        B, _, H, W = ir.shape
        
        # 1. 输入投影: IR + RGB -> Quaternion
        q_input = self.input_proj(ir, rgb)  # [B, C, H, W]
        
        # 转换为序列格式用于 Q-SSM
        # [B, C, H, W] -> [B, H*W, C]
        q_seq = self._to_sequence(q_input)
        
        # 2. 编码器
        skip_connections = []
        x = q_seq
        
        for i, (blocks, downsample) in enumerate(
            zip(self.encoder_layers, self.downsample_layers)
        ):
            # Q-Mamba blocks
            for block in blocks:
                x = block(x)
            
            skip_connections.append(x)
            
            # 转回图像格式进行下采样
            H_cur, W_cur = H // (2 ** i), W // (2 ** i)
            x = self._to_image(x, H_cur, W_cur)
            x = downsample(x)
            
            # 转回序列格式
            x = self._to_sequence(x)
        
        # 3. Bottleneck
        for block in self.bottleneck:
            x = block(x)
        
        # 4. 融合模块
        H_bot = H // (2 ** (self.num_layers - 1))
        W_bot = W // (2 ** (self.num_layers - 1))
        x = self._to_image(x, H_bot, W_bot)
        x = self.fusion(x)
        x = self._to_sequence(x)
        
        # 5. 解码器
        for i, (upsample, blocks) in enumerate(
            zip(self.upsample_layers, self.decoder_layers)
        ):
            # 上采样
            H_cur = H_bot * (2 ** (i + 1))
            W_cur = W_bot * (2 ** (i + 1))
            x = self._to_image(x, H_bot * (2 ** i), W_bot * (2 ** i))
            x = upsample(x)
            
            # Skip connection
            skip = skip_connections[-(i+1)]
            x = self._to_sequence(x)
            x = x + skip
            
            # Q-Mamba blocks
            for block in blocks:
                x = block(x)
        
        # 6. 输出投影
        x = self._to_image(x, H, W)
        x = self.output_proj(x)
        
        # 转换为实数张量
        x_real = x.to_real_tensor()  # [B, 4C, H, W]
        
        # 最终卷积
        fused = self.final_conv(x_real)  # [B, 3, H, W]
        
        # Sigmoid 确保输出在 [0, 1]
        fused = torch.sigmoid(fused)
        
        return fused
    
    def _to_sequence(self, q_image: QuaternionTensor) -> QuaternionTensor:
        """
        [B, C, H, W] -> [B, H*W, C]
        """
        r = rearrange(q_image.r, 'b c h w -> b (h w) c')
        i = rearrange(q_image.i, 'b c h w -> b (h w) c')
        j = rearrange(q_image.j, 'b c h w -> b (h w) c')
        k = rearrange(q_image.k, 'b c h w -> b (h w) c')
        return QuaternionTensor(r, i, j, k)
    
    def _to_image(self, q_seq: QuaternionTensor, H: int, W: int) -> QuaternionTensor:
        """
        [B, H*W, C] -> [B, C, H, W]
        """
        r = rearrange(q_seq.r, 'b (h w) c -> b c h w', h=H, w=W)
        i = rearrange(q_seq.i, 'b (h w) c -> b c h w', h=H, w=W)
        j = rearrange(q_seq.j, 'b (h w) c -> b c h w', h=H, w=W)
        k = rearrange(q_seq.k, 'b (h w) c -> b c h w', h=H, w=W)
        return QuaternionTensor(r, i, j, k)