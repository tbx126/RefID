import torch
import torch.nn as nn

class HardAttention(nn.Module):
    """硬注意力模块：使用索引从V中选择最相关的补丁"""
    def __init__(self):
        super().__init__()
        
    def forward(self, V, H):
        """
        参数:
            V: 值特征 [B, C, H, W]
            H: 硬注意力索引 [B, HW]，表示每个位置最相关的索引
        返回:
            P: 硬注意力输出 [B, C, H, W]
        """
        B, C, H_dim, W_dim = V.shape
        HW = H_dim * W_dim
        
        # 将V展平为 [B, C, HW]
        V_flat = V.view(B, C, -1)
        
        # 高效地使用gather操作来选择特征
        # 扩展索引以匹配通道维度
        H_expanded = H.unsqueeze(1).expand(-1, C, -1)
        
        # 使用gather操作一次性选择所有特征
        P_flat = torch.gather(V_flat, 2, H_expanded)
        
        # 恢复原始形状
        P = P_flat.view(B, C, H_dim, W_dim)
        
        return P