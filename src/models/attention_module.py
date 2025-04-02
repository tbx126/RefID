import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionEmbedding(nn.Module):
    """注意力嵌入模块，采用分块计算以节省内存"""
    def __init__(self, scale_factor=1.0, chunk_size=1024):
        super().__init__()
        self.scale_factor = scale_factor
        self.chunk_size = chunk_size
        
    def forward(self, Q, K, V):
        """
        参数:
            Q: 查询特征 [B, C, H, W]
            K: 键特征 [B, C, H, W]
            V: 值特征 [B, C, H, W]
        返回:
            S: 软注意力分数矩阵 [B, HW, HW]的一小部分，用于可视化
            H: 硬注意力索引 [B, HW]，表示每个查询位置最相关的键位置索引
        """
        B, C, H, W = Q.shape
        HW = H * W
        
        # 调整形状用于矩阵乘法
        Q_flat = Q.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        K_flat = K.view(B, C, -1)  # [B, C, HW]
        
        # 分块计算注意力并只保存最大索引
        H = torch.zeros(B, HW, dtype=torch.long, device=Q.device)
        
        # 为可视化保存一个小块S
        vis_size = min(1024, HW)
        S_vis = torch.zeros(B, vis_size, vis_size, device=Q.device)
        
        for i in range(0, HW, self.chunk_size):
            end_i = min(i + self.chunk_size, HW)
            chunk_size_i = end_i - i
            
            # 为当前块计算注意力分数
            Q_chunk = Q_flat[:, i:end_i, :]  # [B, chunk_size, C]
            
            max_vals = torch.full((B, chunk_size_i), -float('inf'), device=Q.device)
            max_idxs = torch.zeros(B, chunk_size_i, dtype=torch.long, device=Q.device)
            
            for j in range(0, HW, self.chunk_size):
                end_j = min(j + self.chunk_size, HW)
                
                # 计算当前块的注意力分数
                K_chunk = K_flat[:, :, j:end_j]  # [B, C, chunk_size]
                S_chunk = torch.matmul(Q_chunk, K_chunk) * self.scale_factor  # [B, chunk_size_i, chunk_size_j]
                
                # 保存一部分S用于可视化
                if i < vis_size and j < vis_size:
                    i_end = min(vis_size, end_i)
                    j_end = min(vis_size, end_j)
                    S_vis[:, i:i_end, j:j_end] = S_chunk[:, :i_end-i, :j_end-j]
                
                # 更新最大值和索引
                vals, idxs = torch.max(S_chunk, dim=2)  # [B, chunk_size_i]
                idxs = idxs + j  # 调整索引以匹配全局位置
                
                # 更新全局最大值
                mask = vals > max_vals
                max_vals[mask] = vals[mask]
                max_idxs[mask] = idxs[mask]
            
            # 存储此块的最大索引
            H[:, i:end_i] = max_idxs
        
        return S_vis, H