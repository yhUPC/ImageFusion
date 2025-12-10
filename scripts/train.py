"""
Training script for Quaternion Mamba Fusion
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
import time
import numpy as np

from quaternion_mamba.models.fusion_model import QuaternionMambaFusion
from quaternion_mamba.losses.fusion_loss import FusionLoss
from quaternion_mamba.data.datasets import get_dataloader
from quaternion_mamba.utils.metrics import MetricEvaluator
from quaternion_mamba.utils.visualization import (
    visualize_fusion_results,
    plot_training_curves,
    save_image
)


class Trainer:
    """训练器"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建目录
        self.checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        self.log_dir = Path(config['paths']['log_dir'])
        self.result_dir = Path(config['paths']['result_dir'])
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # 构建模型
        self.model = QuaternionMambaFusion(
            img_size=config['data']['image_size'],
            embed_dim=config['model']['d_model'],
            depths=[2, 2, 4, 2],
            d_state=config['model']['d_state'],
            dropout=config['model']['dropout'],
        ).to(self.device)
        
        print(f"Model built. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 损失函数
        self.criterion = FusionLoss(
            ssim_weight=config['loss']['ssim_weight'],
            gradient_weight=config['loss']['texture_weight'],
            intensity_weight=config['loss']['intensity_weight'],
            perceptual_weight=config['loss']['perceptual_weight'],
        ).to(self.device)
        
        # 优化器
        if config['train']['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['train']['lr'],
                weight_decay=config['train']['weight_decay'],
                betas=config['train']['betas'],
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['train']['lr'],
            )
        
        # 学习率调度器
        if config['train']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['train']['epochs'],
                eta_min=config['train']['min_lr'],
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=50,
                gamma=0.5,
            )
        
        # 数据加载器
        self.train_loader = get_dataloader(
            root_dir=config['paths']['data_root'],
            dataset_name=config['data']['train_datasets'][0],
            split='train',
            batch_size=config['train']['batch_size'],
            num_workers=config['train']['num_workers'],
            img_size=config['data']['image_size'],
            shuffle=True,
        )
        
        # 验证数据加载器
        try:
            self.val_loader = get_dataloader(
                root_dir=config['paths']['data_root'],
                dataset_name=config['data']['train_datasets'][0],
                split='val',
                batch_size=config['train']['batch_size'],
                num_workers=config['train']['num_workers'],
                img_size=config['data']['image_size'],
                shuffle=False,
            )
        except:
            print("No validation set found, using test set for validation")
            self.val_loader = get_dataloader(
                root_dir=config['paths']['data_root'],
                dataset_name=config['data']['test_datasets'][0],
                split='test',
                batch_size=1,
                num_workers=config['train']['num_workers'],
                img_size=config['data']['image_size'],
                shuffle=False,
            )
        
        # 评估器
        self.evaluator = MetricEvaluator()
        
        # 训练状态
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'ssim_loss': [],
            'gradient_loss': [],
            'intensity_loss': [],
        }
    
    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        
        epoch_loss = 0
        epoch_loss_dict = {
            'loss_ssim': 0,
            'loss_gradient': 0,
            'loss_intensity': 0,
            'loss_perceptual': 0,
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["train"]["epochs"]}')
        
        for batch_idx, (ir, rgb, _) in enumerate(pbar):
            ir = ir.to(self.device)
            rgb = rgb.to(self.device)
            
            # 前向传播
            fused = self.model(ir, rgb)
            
            # 计算损失
            loss, loss_dict = self.criterion(fused, ir, rgb)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config['train']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['train']['grad_clip']
                )
            
            self.optimizer.step()
            
            # 记录
            epoch_loss += loss.item()
            for key in epoch_loss_dict:
                epoch_loss_dict[key] += loss_dict[key]
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
            })
            
            # 记录到 TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/loss_total', loss.item(), global_step)
            for key, value in loss_dict.items():
                self.writer.add_scalar(f'Train/{key}', value, global_step)
        
        # 平均损失
        epoch_loss /= len(self.train_loader)
        for key in epoch_loss_dict:
            epoch_loss_dict[key] /= len(self.train_loader)
        
        return epoch_loss, epoch_loss_dict
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        
        val_loss = 0
        val_metrics = {
            'EN': 0, 'MI': 0, 'SF': 0, 'AG': 0,
            'SD': 0, 'SSIM': 0, 'Q_ABF': 0
        }
        
        pbar = tqdm(self.val_loader, desc='Validating')
        
        for batch_idx, (ir, rgb, names) in enumerate(pbar):
            ir = ir.to(self.device)
            rgb = rgb.to(self.device)
            
            # 前向传播
            fused = self.model(ir, rgb)
            
            # 计算损失
            loss, _ = self.criterion(fused, ir, rgb)
            val_loss += loss.item()
            
            # 计算指标 (仅对第一个样本)
            if batch_idx < 10:
                ir_np = ir[0].cpu().squeeze().numpy()
                rgb_np = rgb[0].cpu().numpy().transpose(1, 2, 0)
                fused_np = fused[0].cpu().numpy().transpose(1, 2, 0)
                
                metrics = self.evaluator.evaluate(fused_np, ir_np, rgb_np)
                
                for key in val_metrics:
                    if key in metrics:
                        val_metrics[key] += metrics[key]
            
            # 保存一些可视化结果
            if batch_idx < 5:
                save_path = self.result_dir / f'epoch_{epoch}_sample_{batch_idx}.png'
                visualize_fusion_results(
                    ir[0].cpu(),
                    rgb[0].cpu(),
                    fused[0].cpu(),
                    save_path=save_path
                )
        
        # 平均
        val_loss /= len(self.val_loader)
        num_samples = min(10, len(self.val_loader))
        for key in val_metrics:
            val_metrics[key] /= num_samples
        
        # 记录到 TensorBoard
        self.writer.add_scalar('Val/loss', val_loss, epoch)
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        return val_loss, val_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'config': self.config,
        }
        
        # 保存最新
        checkpoint_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # 保存最佳
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Best checkpoint saved to {best_path}")
        
        # 每 50 个 epoch 保存一次
        if epoch % 50 == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found")
            return
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
        
        print(f"Resumed from epoch {self.start_epoch}")
    
    def train(self):
        """主训练循环"""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['train']['epochs']}")
        print(f"Batch size: {self.config['train']['batch_size']}")
        print(f"Learning rate: {self.config['train']['lr']}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("="*50 + "\n")
        
        for epoch in range(self.start_epoch, self.config['train']['epochs']):
            start_time = time.time()
            
            # 训练
            train_loss, train_loss_dict = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_metrics = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['ssim_loss'].append(train_loss_dict['loss_ssim'])
            self.history['gradient_loss'].append(train_loss_dict['loss_gradient'])
            self.history['intensity_loss'].append(train_loss_dict['loss_intensity'])
            
            # 打印
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val EN:     {val_metrics['EN']:.4f}")
            print(f"  Val MI:     {val_metrics['MI']:.4f}")
            print(f"  Val SSIM:   {val_metrics['SSIM']:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存检查点
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            self.save_checkpoint(epoch, is_best=is_best)
            
            # 绘制训练曲线
            if epoch % 10 == 0:
                plot_path = self.result_dir / 'training_curves.png'
                plot_training_curves(self.history, save_path=plot_path)
        
        print("\n" + "="*50)
        print("Training Completed!")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print("="*50 + "\n")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Quaternion Mamba Fusion')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 加载检查点（如果有）
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 训练
    trainer.train()


if __name__ == '__main__':
    main()