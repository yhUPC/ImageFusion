"""
Dataset loaders for multi-modal image fusion
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, List, Optional
import glob


class FusionDataset(Dataset):
    """
    通用融合数据集
    """
    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        split: str = 'train',
        transform=None,
        img_size: int = 256,
    ):
        """
        Args:
            root_dir: 数据根目录
            dataset_name: 数据集名称 (TNO, MSRS, RoadScene, M3FD)
            split: train/val/test
            transform: 数据增强
            img_size: 图像大小
        """
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.img_size = img_size
        
        # 构建数据路径
        self.data_dir = os.path.join(root_dir, dataset_name, split)
        
        # 获取图像对
        self.ir_images = sorted(glob.glob(os.path.join(self.data_dir, 'ir', '*.png')) +
                                glob.glob(os.path.join(self.data_dir, 'ir', '*.jpg')))
        self.vis_images = sorted(glob.glob(os.path.join(self.data_dir, 'vi', '*.png')) +
                                 glob.glob(os.path.join(self.data_dir, 'vi', '*.jpg')))
        
        assert len(self.ir_images) == len(self.vis_images), \
            f"IR and VIS image counts mismatch: {len(self.ir_images)} vs {len(self.vis_images)}"
        
        print(f"Loaded {len(self.ir_images)} image pairs from {dataset_name}/{split}")
    
    def __len__(self):
        return len(self.ir_images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            ir: [1, H, W] 红外图像
            rgb: [3, H, W] 可见光图像
            name: 图像名称
        """
        # 读取图像
        ir_path = self.ir_images[idx]
        vis_path = self.vis_images[idx]
        
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        vis = cv2.imread(vis_path, cv2.IMREAD_COLOR)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        
        # Resize
        ir = cv2.resize(ir, (self.img_size, self.img_size))
        vis = cv2.resize(vis, (self.img_size, self.img_size))
        
        # 转换为张量
        ir = torch.from_numpy(ir).float() / 255.0
        ir = ir.unsqueeze(0)  # [1, H, W]
        
        vis = torch.from_numpy(vis).float() / 255.0
        vis = vis.permute(2, 0, 1)  # [3, H, W]
        
        # 数据增强
        if self.transform is not None:
            ir, vis = self.transform(ir, vis)
        
        # 获取图像名称
        name = os.path.basename(ir_path).split('.')[0]
        
        return ir, vis, name


class MSRSDataset(FusionDataset):
    """
    MSRS 数据集 (带分割标签)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name='MSRS', **kwargs)
        
        # 加载分割标签（如果有）
        self.seg_dir = os.path.join(self.data_dir, 'labels')
        if os.path.exists(self.seg_dir):
            self.seg_images = sorted(glob.glob(os.path.join(self.seg_dir, '*.png')))
            self.has_labels = True
        else:
            self.has_labels = False
    
    def __getitem__(self, idx):
        ir, vis, name = super().__getitem__(idx)
        
        if self.has_labels:
            seg_path = self.seg_images[idx]
            seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            seg = cv2.resize(seg, (self.img_size, self.img_size), 
                           interpolation=cv2.INTER_NEAREST)
            seg = torch.from_numpy(seg).long()
            return ir, vis, seg, name
        
        return ir, vis, name


class TNODataset(FusionDataset):
    """
    TNO 数据集
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name='TNO', **kwargs)


class RoadSceneDataset(FusionDataset):
    """
    RoadScene 数据集
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name='RoadScene', **kwargs)


def get_dataloader(
    root_dir: str,
    dataset_name: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    img_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    """
    获取数据加载器
    """
    # 选择数据集
    if dataset_name == 'MSRS':
        dataset = MSRSDataset(root_dir, split=split, img_size=img_size)
    elif dataset_name == 'TNO':
        dataset = TNODataset(root_dir, split=split, img_size=img_size)
    elif dataset_name == 'RoadScene':
        dataset = RoadSceneDataset(root_dir, split=split, img_size=img_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
    
    return dataloader