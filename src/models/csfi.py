import torch
import torch.nn as nn

class CSFI(nn.Module):
    """跨尺度特征集成模块：集成软注意力输出和原始特征"""
    def __init__(self, in_channels):
        super().__init__()
        
        # 特征转换层
        self.conv_soft = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_orig = nn.Conv2d(3, in_channels, kernel_size=3, padding=1)  # 原始图像为3通道
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, soft_output, original_img):
        """
        参数:
            soft_output: 软注意力输出 [B, C, H, W]
            original_img: 原始图像 [B, 3, H, W]
        返回:
            output: 融合后的特征 [B, C, H, W]
        """
        # 转换特征
        soft_features = self.conv_soft(soft_output)
        orig_features = self.conv_orig(original_img)
        
        # 拼接特征
        combined = torch.cat([soft_features, orig_features], dim=1)
        
        # 融合
        output = self.fusion(combined)
        
        return output