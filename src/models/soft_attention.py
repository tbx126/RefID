import torch
import torch.nn as nn

class SoftAttention(nn.Module):
    """超简化版软注意力模块"""
    def __init__(self, chunk_size=1024):
        super().__init__()
        # 不再需要分块大小
        
        # 使用标准卷积处理
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
    def forward(self, P, S=None):
        """
        参数:
            P: 硬注意力输出 [B, C, H, W]
            S: 不使用
        返回:
            soft_output: 软注意力输出 [B, C, H, W]
        """
        # 简单卷积处理
        x = self.conv1(P)
        x = self.relu(x)
        x = self.conv2(x)
        
        # 残差连接
        soft_output = P + x
        
        return soft_output