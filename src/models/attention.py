# src/models/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionEmbedding(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, Q, K, V):
        B, C, H, W = Q.shape
        Q_flat = Q.view(B, C, -1).permute(0, 2, 1)
        K_flat = K.view(B, C, -1).permute(0, 2, 1)
        V_flat = V.view(B, C, -1).permute(0, 2, 1)
        
        # 线性变换
        q = self.q_proj(Q_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(K_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(V_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        S = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(S, v)
        
        # 输出处理
        attn_output = attn_output.transpose(1, 2).reshape(B, H*W, C)
        H_out = self.out_proj(attn_output).permute(0, 2, 1).view(B, C, H, W)
        
        return S, H_out