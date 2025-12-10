"""
Evaluation metrics for image fusion
图像融合的评估指标
"""

import torch
import numpy as np
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as compare_ssim
import cv2


def entropy(image):
    """
    信息熵 (Entropy, EN)
    衡量图像的信息量
    """
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    # 转换为灰度图
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 计算直方图
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 1))
    hist = hist / hist.sum()
    
    # 计算熵
    hist = hist[hist > 0]  # 移除零值
    en = -np.sum(hist * np.log2(hist))
    
    return en


def mutual_information(image1, image2):
    """
    互信息 (Mutual Information, MI)
    衡量融合图像包含源图像的信息量
    """
    if torch.is_tensor(image1):
        image1 = image1.cpu().numpy()
    if torch.is_tensor(image2):
        image2 = image2.cpu().numpy()
    
    # 转换为灰度图
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    # 量化到 0-255
    image1 = (image1 * 255).astype(np.uint8)
    image2 = (image2 * 255).astype(np.uint8)
    
    # 计算联合直方图
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=256)
    
    # 归一化
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    # 计算互信息
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))
    
    return mi


def spatial_frequency(image):
    """
    空间频率 (Spatial Frequency, SF)
    衡量图像的整体活跃度
    """
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 行频率
    RF = np.sqrt(np.mean((image[:, 1:] - image[:, :-1])**2))
    
    # 列频率
    CF = np.sqrt(np.mean((image[1:, :] - image[:-1, :])**2))
    
    # 空间频率
    sf = np.sqrt(RF**2 + CF**2)
    
    return sf


def average_gradient(image):
    """
    平均梯度 (Average Gradient, AG)
    衡量图像的清晰度
    """
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Sobel 算子
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # 梯度幅值
    grad = np.sqrt(sobelx**2 + sobely**2)
    
    # 平均梯度
    ag = np.mean(grad)
    
    return ag


def standard_deviation(image):
    """
    标准差 (Standard Deviation, SD)
    衡量图像的对比度
    """
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    return np.std(image)


def correlation_coefficient(image1, image2):
    """
    相关系数 (Correlation Coefficient, CC)
    衡量融合图像与源图像的相似度
    """
    if torch.is_tensor(image1):
        image1 = image1.cpu().numpy()
    if torch.is_tensor(image2):
        image2 = image2.cpu().numpy()
    
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    # 确保相同尺寸
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    # 计算相关系数
    img1_flat = image1.flatten()
    img2_flat = image2.flatten()
    
    cc = np.corrcoef(img1_flat, img2_flat)[0, 1]
    
    return cc


def ssim(image1, image2):
    """
    结构相似性 (Structural Similarity, SSIM)
    """
    if torch.is_tensor(image1):
        image1 = image1.cpu().numpy()
    if torch.is_tensor(image2):
        image2 = image2.cpu().numpy()
    
    # 确保范围在 [0, 1]
    image1 = np.clip(image1, 0, 1)
    image2 = np.clip(image2, 0, 1)
    
    # 计算 SSIM
    if len(image1.shape) == 3:
        ssim_val = compare_ssim(image1, image2, multichannel=True, data_range=1.0)
    else:
        ssim_val = compare_ssim(image1, image2, data_range=1.0)
    
    return ssim_val


def q_abf(image_f, image_a, image_b):
    """
    Q_ABF 指标 - 基于梯度的融合质量评估
    
    Args:
        image_f: 融合图像
        image_a: 源图像A (IR)
        image_b: 源图像B (RGB)
    """
    if torch.is_tensor(image_f):
        image_f = image_f.cpu().numpy()
    if torch.is_tensor(image_a):
        image_a = image_a.cpu().numpy()
    if torch.is_tensor(image_b):
        image_b = image_b.cpu().numpy()
    
    # 转灰度图
    if len(image_f.shape) == 3:
        image_f = cv2.cvtColor(image_f, cv2.COLOR_RGB2GRAY)
    if len(image_a.shape) == 3:
        image_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY)
    if len(image_b.shape) == 3:
        image_b = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY)
    
    # 计算梯度
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    gf_x = convolve2d(image_f, sobelx, mode='same', boundary='symm')
    gf_y = convolve2d(image_f, sobely, mode='same', boundary='symm')
    gf = np.sqrt(gf_x**2 + gf_y**2)
    
    ga_x = convolve2d(image_a, sobelx, mode='same', boundary='symm')
    ga_y = convolve2d(image_a, sobely, mode='same', boundary='symm')
    ga = np.sqrt(ga_x**2 + ga_y**2)
    
    gb_x = convolve2d(image_b, sobelx, mode='same', boundary='symm')
    gb_y = convolve2d(image_b, sobely, mode='same', boundary='symm')
    gb = np.sqrt(gb_x**2 + gb_y**2)
    
    # 计算权重
    eps = 1e-10
    w_a = ga / (ga + gb + eps)
    w_b = gb / (ga + gb + eps)
    
    # 计算 Q_ABF
    qaf = (w_a * gf * ga + w_b * gf * gb) / (w_a * ga + w_b * gb + eps)
    
    return np.mean(qaf)


class MetricEvaluator:
    """
    融合指标评估器
    """
    def __init__(self):
        self.metrics = {
            'EN': entropy,
            'MI': mutual_information,
            'SF': spatial_frequency,
            'AG': average_gradient,
            'SD': standard_deviation,
            'CC': correlation_coefficient,
            'SSIM': ssim,
            'Q_ABF': q_abf,
        }
    
    def evaluate(self, fused, ir, rgb):
        """
        评估所有指标
        
        Args:
            fused: [H, W, 3] 或 [H, W] numpy 数组
            ir: [H, W] numpy 数组
            rgb: [H, W, 3] numpy 数组
        
        Returns:
            dict: 所有指标的结果
        """
        results = {}
        
        # 单图像指标
        results['EN'] = self.metrics['EN'](fused)
        results['SF'] = self.metrics['SF'](fused)
        results['AG'] = self.metrics['AG'](fused)
        results['SD'] = self.metrics['SD'](fused)
        
        # 双图像指标
        results['MI_IR'] = self.metrics['MI'](fused, ir)
        results['MI_RGB'] = self.metrics['MI'](fused, rgb)
        results['MI'] = (results['MI_IR'] + results['MI_RGB']) / 2
        
        results['CC_IR'] = self.metrics['CC'](fused, ir)
        results['CC_RGB'] = self.metrics['CC'](fused, rgb)
        results['CC'] = (results['CC_IR'] + results['CC_RGB']) / 2
        
        results['SSIM_IR'] = self.metrics['SSIM'](fused, ir if len(ir.shape) == 2 else cv2.cvtColor(ir, cv2.COLOR_RGB2GRAY))
        results['SSIM_RGB'] = self.metrics['SSIM'](fused, rgb)
        results['SSIM'] = (results['SSIM_IR'] + results['SSIM_RGB']) / 2
        
        # Q_ABF
        results['Q_ABF'] = self.metrics['Q_ABF'](fused, ir, rgb)
        
        return results
    
    def print_results(self, results):
        """打印结果"""
        print("\n" + "="*50)
        print("Fusion Metrics Evaluation")
        print("="*50)
        print(f"EN    (Entropy):              {results['EN']:.4f}")
        print(f"MI    (Mutual Information):   {results['MI']:.4f}")
        print(f"SF    (Spatial Frequency):    {results['SF']:.4f}")
        print(f"AG    (Average Gradient):     {results['AG']:.4f}")
        print(f"SD    (Standard Deviation):   {results['SD']:.4f}")
        print(f"CC    (Correlation Coef.):    {results['CC']:.4f}")
        print(f"SSIM  (Structural Similarity):{results['SSIM']:.4f}")
        print(f"Q_ABF (Gradient-based):       {results['Q_ABF']:.4f}")
        print("="*50 + "\n")