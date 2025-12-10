"""
Testing script for Quaternion Mamba Fusion
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import time

from quaternion_mamba.models.fusion_model import QuaternionMambaFusion
from quaternion_mamba.data.datasets import get_dataloader
from quaternion_mamba.utils.metrics import MetricEvaluator
from quaternion_mamba.utils.visualization import (
    visualize_fusion_results,
    visualize_quaternion_features,
    save_image
)


class Tester:
    """测试器"""
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建结果目录
        self.result_dir = Path(config['paths']['result_dir'])
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建模型
        self.model = QuaternionMambaFusion(
            img_size=config['data']['image_size'],
            embed_dim=config['model']['d_model'],
            depths=[2, 2, 4, 2],
            d_state=config['model']['d_state'],
            dropout=0.0,  # 测试时不使用 dropout
        ).to(self.device)
        
        # 加载权重
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 评估器
        self.evaluator = MetricEvaluator()
    
    @torch.no_grad()
    def test_dataset(self, dataset_name):
        """测试一个数据集"""
        print(f"\n{'='*50}")
        print(f"Testing on {dataset_name}")
        print(f"{'='*50}")
        
        # 数据加载器
        test_loader = get_dataloader(
            root_dir=self.config['paths']['data_root'],
            dataset_name=dataset_name,
            split='test',
            batch_size=1,
            num_workers=4,
            img_size=self.config['data']['image_size'],
            shuffle=False,
        )
        
        # 创建数据集结果目录
        dataset_result_dir = self.result_dir / dataset_name
        dataset_result_dir.mkdir(exist_ok=True)
        
        # 记录所有结果
        all_results = []
        inference_times = []
        
        pbar = tqdm(test_loader, desc=f'Testing {dataset_name}')
        
        for batch_idx, (ir, rgb, names) in enumerate(pbar):
            ir = ir.to(self.device)
            rgb = rgb.to(self.device)
            
            # 推理计时
            torch.cuda.synchronize()
            start_time = time.time()
            
            fused = self.model(ir, rgb)
            
            torch.cuda.synchronize()
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 转换为 numpy
            ir_np = ir[0].cpu().squeeze().numpy()
            rgb_np = rgb[0].cpu().numpy().transpose(1, 2, 0)
            fused_np = fused[0].cpu().numpy().transpose(1, 2, 0)
            
            # 计算指标
            metrics = self.evaluator.evaluate(fused_np, ir_np, rgb_np)
            metrics['name'] = names[0]
            metrics['inference_time'] = inference_time
            all_results.append(metrics)
            
            # 保存融合图像
            fused_path = dataset_result_dir / f'{names[0]}_fused.png'
            save_image(fused_np, fused_path)
            
            # 保存可视化
            vis_path = dataset_result_dir / f'{names[0]}_comparison.png'
            visualize_fusion_results(ir_np, rgb_np, fused_np, save_path=vis_path)
            
            # 更新进度条
            pbar.set_postfix({
                'EN': f'{metrics["EN"]:.2f}',
                'MI': f'{metrics["MI"]:.2f}',
                'SSIM': f'{metrics["SSIM"]:.3f}',
                'Time': f'{inference_time:.3f}s'
            })
        
        # 统计结果
        df = pd.DataFrame(all_results)
        
        # 计算平均值
        mean_metrics = df.mean(numeric_only=True)
        std_metrics = df.std(numeric_only=True)
        
        # 保存详细结果
        csv_path = dataset_result_dir / 'detailed_results.csv'
        df.to_csv(csv_path, index=False)
        
        # 打印统计
        print(f"\n{'='*50}")
        print(f"Results on {dataset_name}")
        print(f"{'='*50}")
        print(f"Number of images: {len(all_results)}")
        print(f"\nMetrics (Mean ± Std):")
        print(f"  EN:    {mean_metrics['EN']:.4f} ± {std_metrics['EN']:.4f}")
        print(f"  MI:    {mean_metrics['MI']:.4f} ± {std_metrics['MI']:.4f}")
        print(f"  SF:    {mean_metrics['SF']:.4f} ± {std_metrics['SF']:.4f}")
        print(f"  AG:    {mean_metrics['AG']:.4f} ± {std_metrics['AG']:.4f}")
        print(f"  SD:    {mean_metrics['SD']:.4f} ± {std_metrics['SD']:.4f}")
        print(f"  CC:    {mean_metrics['CC']:.4f} ± {std_metrics['CC']:.4f}")
        print(f"  SSIM:  {mean_metrics['SSIM']:.4f} ± {std_metrics['SSIM']:.4f}")
        print(f"  Q_ABF: {mean_metrics['Q_ABF']:.4f} ± {std_metrics['Q_ABF']:.4f}")
        print(f"\nInference Time:")
        print(f"  Mean: {np.mean(inference_times):.4f}s")
        print(f"  Std:  {np.std(inference_times):.4f}s")
        print(f"  FPS:  {1.0/np.mean(inference_times):.2f}")
        print(f"{'='*50}\n")
        
        # 保存统计结果
        summary_path = dataset_result_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Results on {dataset_name}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Number of images: {len(all_results)}\n\n")
            f.write("Metrics (Mean ± Std):\n")
            f.write(f"  EN:    {mean_metrics['EN']:.4f} ± {std_metrics['EN']:.4f}\n")
            f.write(f"  MI:    {mean_metrics['MI']:.4f} ± {std_metrics['MI']:.4f}\n")
            f.write(f"  SF:    {mean_metrics['SF']:.4f} ± {std_metrics['SF']:.4f}\n")
            f.write(f"  AG:    {mean_metrics['AG']:.4f} ± {std_metrics['AG']:.4f}\n")
            f.write(f"  SD:    {mean_metrics['SD']:.4f} ± {std_metrics['SD']:.4f}\n")
            f.write(f"  CC:    {mean_metrics['CC']:.4f} ± {std_metrics['CC']:.4f}\n")
            f.write(f"  SSIM:  {mean_metrics['SSIM']:.4f} ± {std_metrics['SSIM']:.4f}\n")
            f.write(f"  Q_ABF: {mean_metrics['Q_ABF']:.4f} ± {std_metrics['Q_ABF']:.4f}\n")
            f.write(f"\nInference Time:\n")
            f.write(f"  Mean: {np.mean(inference_times):.4f}s\n")
            f.write(f"  FPS:  {1.0/np.mean(inference_times):.2f}\n")
        
        return mean_metrics
    
    def test_all(self):
        """测试所有数据集"""
        all_datasets_results = {}
        
        for dataset_name in self.config['data']['test_datasets']:
            try:
                mean_metrics = self.test_dataset(dataset_name)
                all_datasets_results[dataset_name] = mean_metrics
            except Exception as e:
                print(f"Error testing {dataset_name}: {e}")
        
        # 汇总所有结果
        print("\n" + "="*70)
        print("Summary of All Datasets")
        print("="*70)
        
        # 创建表格
        metrics_names = ['EN', 'MI', 'SF', 'AG', 'SD', 'CC', 'SSIM', 'Q_ABF']
        
        print(f"{'Dataset':<15}", end='')
        for metric in metrics_names:
            print(f"{metric:>8}", end='')
        print()
        print("-"*70)
        
        for dataset_name, metrics in all_datasets_results.items():
            print(f"{dataset_name:<15}", end='')
            for metric in metrics_names:
                print(f"{metrics[metric]:>8.4f}", end='')
            print()
        
        print("="*70 + "\n")
        
        # 保存汇总结果
        summary_df = pd.DataFrame(all_datasets_results).T
        summary_path = self.result_dir / 'all_datasets_summary.csv'
        summary_df.to_csv(summary_path)
        print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Quaternion Mamba Fusion')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset to test (if None, test all)')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建测试器
    tester = Tester(config, args.checkpoint)
    
    # 测试
    if args.dataset:
        tester.test_dataset(args.dataset)
    else:
        tester.test_all()


if __name__ == '__main__':
    main()