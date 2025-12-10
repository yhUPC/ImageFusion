"""
Data augmentation for fusion
"""

import torch
import torch.nn.functional as F
import random


class RandomFlip:
    """随机翻转"""
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, ir, vis):
        if random.random() < self.prob:
            # 水平翻转
            ir = torch.flip(ir, dims=[-1])
            vis = torch.flip(vis, dims=[-1])
        
        if random.random() < self.prob:
            # 垂直翻转
            ir = torch.flip(ir, dims=[-2])
            vis = torch.flip(vis, dims=[-2])
        
        return ir, vis


class RandomRotate:
    """随机旋转 90度"""
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, ir, vis):
        if random.random() < self.prob:
            k = random.randint(1, 3)  # 旋转 90, 180, 270 度
            ir = torch.rot90(ir, k, dims=[-2, -1])
            vis = torch.rot90(vis, k, dims=[-2, -1])
        
        return ir, vis


class RandomCrop:
    """随机裁剪"""
    def __init__(self, size: int):
        self.size = size
    
    def __call__(self, ir, vis):
        _, H, W = ir.shape
        
        if H > self.size and W > self.size:
            top = random.randint(0, H - self.size)
            left = random.randint(0, W - self.size)
            
            ir = ir[:, top:top+self.size, left:left+self.size]
            vis = vis[:, top:top+self.size, left:left+self.size]
        
        return ir, vis


class Compose:
    """组合多个变换"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, ir, vis):
        for t in self.transforms:
            ir, vis = t(ir, vis)
        return ir, vis


def get_training_augmentation():
    """获取训练时的数据增强"""
    return Compose([
        RandomFlip(prob=0.5),
        RandomRotate(prob=0.5),
    ])