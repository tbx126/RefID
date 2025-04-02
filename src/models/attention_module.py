import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionEmbedding(nn.Module):
    """注意力嵌入模块，将Q和K转换为注意力分数S和注意力输出H"""
    def __init__(self, scale_factor=1.0):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, Q, K, V):
        """
        参数:
            Q: 查询特征 [B, C, H, W]
            K: 键特征 [B, C, H, W]
            V: 值特征 [B, C, H, W]
        返回:
            S: 注意力分数矩阵 [B, HW, HW]
            H: 注意力嵌入输出 [B, C, H, W]
        """
        B, C, H, W = Q.shape
        
        # 调整形状用于矩阵乘法
        Q_flat = Q.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        K_flat = K.view(B, C, -1)  # [B, C, HW]
        V_flat = V.view(B, C, -1)  # [B, C, HW]
        
        # 计算注意力分数
        S = torch.matmul(Q_flat, K_flat) * self.scale_factor  # [B, HW, HW]
        
        # 应用softmax获得权重
        attention_weights = F.softmax(S, dim=-1)  # [B, HW, HW]
        
        # 计算注意力输出
        H_flat = torch.matmul(V_flat, attention_weights.permute(0, 2, 1))  # [B, C, HW]
        H = H_flat.view(B, C, H, W)  # 恢复原始形状 [B, C, H, W]
        
        return S, H