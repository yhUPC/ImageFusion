"""
Script to download and organize fusion datasets
"""

import os
import argparse
from pathlib import Path
import urllib.request
import zipfile
import shutil
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """进度条下载器"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """下载文件"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_tno_dataset(data_root):
    """
    下载 TNO 数据集
    """
    print("\n" + "="*50)
    print("Downloading TNO Dataset")
    print("="*50)
    
    tno_dir = Path(data_root) / 'TNO'
    tno_dir.mkdir(parents=True, exist_ok=True)
    
    # TNO 数据集通常需要手动下载
    # 这里提供组织结构
    print("\nTNO Dataset should be organized as:")
    print(f"{tno_dir}/")
    print("  ├── train/")
    print("  │   ├── ir/")
    print("  │   └── vi/")
    print("  └── test/")
    print("      ├── ir/")
    print("      └── vi/")
    print("\nPlease download from: https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029")
    print(f"And organize the files into {tno_dir}")


def download_msrs_dataset(data_root):
    """
    下载 MSRS 数据集
    """
    print("\n" + "="*50)
    print("Downloading MSRS Dataset")
    print("="*50)
    
    msrs_dir = Path(data_root) / 'MSRS'
    msrs_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nMSRS Dataset should be organized as:")
    print(f"{msrs_dir}/")
    print("  ├── train/")
    print("  │   ├── ir/")
    print("  │   ├── vi/")
    print("  │   └── labels/ (optional)")
    print("  └── test/")
    print("      ├── ir/")
    print("      ├── vi/")
    print("      └── labels/ (optional)")
    print("\nPlease download from: https://github.com/Linfeng-Tang/MSRS")
    print(f"And organize the files into {msrs_dir}")


def download_roadscene_dataset(data_root):
    """
    下载 RoadScene 数据集
    """
    print("\n" + "="*50)
    print("Downloading RoadScene Dataset")
    print("="*50)
    
    roadscene_dir = Path(data_root) / 'RoadScene'
    roadscene_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nRoadScene Dataset should be organized as:")
    print(f"{roadscene_dir}/")
    print("  └── test/")
    print("      ├── ir/")
    print("      └── vi/")
    print("\nPlease download from: https://github.com/hanna-xu/RoadScene")
    print(f"And organize the files into {roadscene_dir}")


def create_dummy_data(data_root):
    """
    创建虚拟数据用于测试
    """
    print("\n" + "="*50)
    print("Creating Dummy Data for Testing")
    print("="*50)
    
    import numpy as np
    from PIL import Image
    
    for dataset_name in ['TNO', 'MSRS', 'RoadScene']:
        dataset_dir = Path(data_root) / dataset_name
        
        for split in ['train', 'test']:
            ir_dir = dataset_dir / split / 'ir'
            vi_dir = dataset_dir / split / 'vi'
            
            ir_dir.mkdir(parents=True, exist_ok=True)
            vi_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建几个虚拟图像
            num_images = 5 if split == 'train' else 3
            
            for i in range(num_images):
                # 红外图像（灰度）
                ir_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
                ir_pil = Image.fromarray(ir_img, mode='L')
                ir_pil.save(ir_dir / f'image_{i:03d}.png')
                
                # 可见光图像（RGB）
                vi_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                vi_pil = Image.fromarray(vi_img, mode='RGB')
                vi_pil.save(vi_dir / f'image_{i:03d}.png')
            
            print(f"Created {num_images} dummy images in {dataset_dir}/{split}")


def main():
    parser = argparse.ArgumentParser(description='Download fusion datasets')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for datasets')
    parser.add_argument('--dataset', type=str, choices=['TNO', 'MSRS', 'RoadScene', 'all'],
                       default='all', help='Which dataset to download')
    parser.add_argument('--dummy', action='store_true',
                       help='Create dummy data for testing')
    args = parser.parse_args()
    
    # 创建数据根目录
    Path(args.data_root).mkdir(parents=True, exist_ok=True)
    
    if args.dummy:
        create_dummy_data(args.data_root)
        return
    
    # 下载数据集
    if args.dataset == 'all' or args.dataset == 'TNO':
        download_tno_dataset(args.data_root)
    
    if args.dataset == 'all' or args.dataset == 'MSRS':
        download_msrs_dataset(args.data_root)
    
    if args.dataset == 'all' or args.dataset == 'RoadScene':
        download_roadscene_dataset(args.data_root)
    
    print("\n" + "="*50)
    print("Download instructions completed!")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()