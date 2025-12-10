"""
Visualization utilities
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path


def save_image(image, path, cmap=None):
    """
    保存图像
    
    Args:
        image: numpy array or torch tensor
        path: 保存路径
        cmap: colormap (for grayscale)
    """
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    # 确保在 [0, 1] 范围内
    image = np.clip(image, 0, 1)
    
    # 转换为 [0, 255]
    image = (image * 255).astype(np.uint8)
    
    # 保存
    if len(image.shape) == 2:
        # 灰度图
        cv2.imwrite(str(path), image)
    else:
        # RGB 图像
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), image)


def visualize_fusion_results(ir, rgb, fused, save_path=None):
    """
    可视化融合结果
    
    Args:
        ir: [1, H, W] or [H, W]
        rgb: [3, H, W] or [H, W, 3]
        fused: [3, H, W] or [H, W, 3]
        save_path: 保存路径
    """
    # 转换为 numpy
    if torch.is_tensor(ir):
        ir = ir.cpu().squeeze().numpy()
    if torch.is_tensor(rgb):
        rgb = rgb.cpu().numpy()
        if rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
    if torch.is_tensor(fused):
        fused = fused.cpu().numpy()
        if fused.shape[0] == 3:
            fused = np.transpose(fused, (1, 2, 0))
    
    # 创建图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(ir, cmap='gray')
    axes[0].set_title('Infrared Image')
    axes[0].axis('off')
    
    axes[1].imshow(rgb)
    axes[1].set_title('Visible Image')
    axes[1].axis('off')
    
    axes[2].imshow(fused)
    axes[2].set_title('Fused Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_quaternion_features(q_tensor, save_path=None):
    """
    可视化四元数特征
    
    Args:
        q_tensor: QuaternionTensor
        save_path: 保存路径
    """
    from ..quaternion.ops import QuaternionTensor
    
    if not isinstance(q_tensor, QuaternionTensor):
        raise ValueError("Input must be QuaternionTensor")
    
    # 获取四个分量
    r = q_tensor.r.cpu().squeeze().numpy()
    i = q_tensor.i.cpu().squeeze().numpy()
    j = q_tensor.j.cpu().squeeze().numpy()
    k = q_tensor.k.cpu().squeeze().numpy()
    
    # 计算模
    norm = q_tensor.norm().cpu().squeeze().numpy()
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(r, cmap='viridis')
    axes[0, 0].set_title('Real Part (r) - IR')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(i, cmap='Reds')
    axes[0, 1].set_title('Imaginary i - R channel')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(j, cmap='Greens')
    axes[0, 2].set_title('Imaginary j - G channel')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(k, cmap='Blues')
    axes[1, 0].set_title('Imaginary k - B channel')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(norm, cmap='hot')
    axes[1, 1].set_title('Quaternion Norm')
    axes[1, 1].axis('off')
    
    # 相位可视化
    phase = np.arctan2(np.sqrt(i**2 + j**2 + k**2), r)
    axes[1, 2].imshow(phase, cmap='twilight')
    axes[1, 2].set_title('Quaternion Phase')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(history, save_path=None):
    """
    绘制训练曲线
    
    Args:
        history: dict with 'epoch', 'train_loss', 'val_loss', etc.
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 总损失
    axes[0, 0].plot(history['epoch'], history['train_loss'], label='Train')
    if 'val_loss' in history:
        axes[0, 0].plot(history['epoch'], history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # SSIM 损失
    if 'ssim_loss' in history:
        axes[0, 1].plot(history['epoch'], history['ssim_loss'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('SSIM Loss')
        axes[0, 1].grid(True)
    
    # 梯度损失
    if 'gradient_loss' in history:
        axes[1, 0].plot(history['epoch'], history['gradient_loss'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Gradient Loss')
        axes[1, 0].grid(True)
    
    # 强度损失
    if 'intensity_loss' in history:
        axes[1, 1].plot(history['epoch'], history['intensity_loss'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Intensity Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()