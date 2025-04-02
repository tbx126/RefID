import torch
import torch.nn as nn
import torch.nn.functional as F

class PSNR:
    """峰值信噪比计算"""
    def __init__(self, max_val=1.0):
        self.max_val = max_val
        
    def __call__(self, pred, target):
        """
        计算PSNR
        Args:
            pred: 预测图像 [B, C, H, W], 范围 [0, 1]
            target: 目标图像 [B, C, H, W], 范围 [0, 1]
        Returns:
            psnr_val: PSNR值 (越高越好)
        """
        assert pred.shape == target.shape, f"预测形状 {pred.shape} 与目标形状 {target.shape} 不匹配"
        
        mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])  # 每个样本的MSE
        psnr_val = 20 * torch.log10(self.max_val / torch.sqrt(mse))  # 每个样本的PSNR
        
        return psnr_val.mean()  # 返回批次平均值

class SSIMLoss(nn.Module):
    """结构相似性损失 (最小化负SSIM)"""
    def __init__(self, window_size=11, sigma=1.5, channels=3, reduction='mean'):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channels = channels
        self.reduction = reduction
        self.register_buffer('window', self._create_window())
        
    def _create_window(self):
        """创建高斯窗口"""
        coords = torch.arange(self.window_size, dtype=torch.float32)
        coords -= self.window_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g /= g.sum()
        
        window = g.unsqueeze(0) @ g.unsqueeze(1)  # 外积得到2D窗口
        window = window.unsqueeze(0).unsqueeze(0).repeat(self.channels, 1, 1, 1)
        
        return window
    
    def _ssim(self, img1, img2):
        """计算SSIM"""
        # 常数，避免除零
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2
        
        # 使用窗口计算均值
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channels)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channels) - mu1_mu2
        
        # SSIM公式
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.reduction == 'none':
            return ssim_map
        elif self.reduction == 'mean':
            return ssim_map.mean()
        else:
            raise ValueError(f"不支持的规约方式: {self.reduction}")
    
    def forward(self, pred, target):
        """
        计算负SSIM损失
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
        Returns:
            loss: 负SSIM值 (越小越好)
        """
        ssim_val = self._ssim(pred, target)
        return 1.0 - ssim_val  # 返回负SSIM

class SSIM:
    """结构相似性评估指标 (用于评估，不用于训练)"""
    def __init__(self, window_size=11, sigma=1.5, channels=3):
        self.loss_fn = SSIMLoss(window_size, sigma, channels)
        
    def __call__(self, pred, target):
        """
        计算SSIM (正值，越高越好)
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
        Returns:
            ssim_val: SSIM值 (越高越好)
        """
        with torch.no_grad():
            # 使用1减去损失值，得到原始SSIM (正值)
            return 1.0 - self.loss_fn(pred, target)

def get_metrics():
    """返回用于评估的指标函数字典"""
    return {
        'psnr': PSNR(),
        'ssim': SSIM()
    }

def get_loss_functions(device):
    """返回用于训练的损失函数字典"""
    return {
        'ssim': SSIMLoss().to(device)
    }