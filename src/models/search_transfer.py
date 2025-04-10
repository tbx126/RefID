import torch
import torch.nn as nn
import torch.nn.functional as F

class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()
        
    def forward(self, dh_img_lv3, dh_ref_lv3, cl_ref_lv1, cl_ref_lv2, cl_ref_lv3):
        """与RefGT对应的特征搜索和转移模块
        
        参数:
            dh_img_lv3: 查询特征 (主要有雾图像的lv3特征)
            dh_ref_lv3: 键特征 (参考有雾图像的lv3特征)
            cl_ref_lv1: 参考清晰图像lv1特征
            cl_ref_lv2: 参考清晰图像lv2特征
            cl_ref_lv3: 参考清晰图像lv3特征
        
        返回:
            S: 相关性图
            T_lv3, T_lv2, T_lv1: 三个层级的转移特征
        """
        k = 1  # topk的k值
        kernel_size = 3
        padding = 1
        
        with torch.no_grad():
            # 搜索阶段
            query_unfold = F.unfold(dh_img_lv3, kernel_size=(kernel_size, kernel_size), padding=padding)
            key_unfold = F.unfold(dh_ref_lv3, kernel_size=(kernel_size, kernel_size), padding=padding)
            key_unfold = key_unfold.permute(0, 2, 1)
            
            # 特征归一化
            key_unfold = F.normalize(key_unfold, dim=2)  # [N, Hr*Wr, C*k*k]
            query_unfold = F.normalize(query_unfold, dim=1)  # [N, C*k*k, H*W]
            
            # 计算相关性
            relevance = torch.bmm(key_unfold, query_unfold)  # [N, Hr*Wr, H*W]
            relevance, relevance_idx = torch.topk(relevance, k=k, dim=1)  # [N, k, H*W]
            
            # 转移阶段
            value_lv3_unfold = F.unfold(cl_ref_lv3, kernel_size=(kernel_size, kernel_size), padding=padding)
            value_lv2_unfold = F.unfold(cl_ref_lv2, kernel_size=(2*kernel_size, 2*kernel_size), 
                                          padding=2*padding, stride=2)
            value_lv1_unfold = F.unfold(cl_ref_lv1, kernel_size=(4*kernel_size, 4*kernel_size), 
                                          padding=4*padding, stride=4)
            
            # 扩展索引
            relevance_lv3_idx = relevance_idx.unsqueeze(1).expand(-1, value_lv3_unfold.size(1), -1, -1).transpose(-2, -1)
            relevance_lv2_idx = relevance_idx.unsqueeze(1).expand(-1, value_lv2_unfold.size(1), -1, -1).transpose(-2, -1)
            relevance_lv1_idx = relevance_idx.unsqueeze(1).expand(-1, value_lv1_unfold.size(1), -1, -1).transpose(-2, -1)
            relevance = relevance.unsqueeze(1).transpose(-2, -1)  # [N, 1, H*W, k]
            
            # 收集值特征
            topk_value_lv3 = value_lv3_unfold.unsqueeze(-2).expand(-1, -1, query_unfold.size(-1), -1).gather(-1, relevance_lv3_idx)
            topk_value_lv2 = value_lv2_unfold.unsqueeze(-2).expand(-1, -1, query_unfold.size(-1), -1).gather(-1, relevance_lv2_idx)
            topk_value_lv1 = value_lv1_unfold.unsqueeze(-2).expand(-1, -1, query_unfold.size(-1), -1).gather(-1, relevance_lv1_idx)
            
            # 加权求和
            T_lv3_unfold = torch.sum(relevance * topk_value_lv3, dim=-1) / torch.sum(relevance, dim=-1)
            T_lv2_unfold = torch.sum(relevance * topk_value_lv2, dim=-1) / torch.sum(relevance, dim=-1)
            T_lv1_unfold = torch.sum(relevance * topk_value_lv1, dim=-1) / torch.sum(relevance, dim=-1)
            
            # 处理重叠
            overlap_cnt_lv3 = F.fold(torch.ones_like(T_lv3_unfold), output_size=dh_img_lv3.size()[-2:],
                                     kernel_size=(kernel_size, kernel_size), padding=padding)
            overlap_cnt_lv2 = F.fold(torch.ones_like(T_lv2_unfold), output_size=(dh_img_lv3.size(2)*2, dh_img_lv3.size(3)*2),
                                     kernel_size=(2*kernel_size, 2*kernel_size), padding=2*padding, stride=2)
            overlap_cnt_lv1 = F.fold(torch.ones_like(T_lv1_unfold), output_size=(dh_img_lv3.size(2)*4, dh_img_lv3.size(3)*4),
                                     kernel_size=(4*kernel_size, 4*kernel_size), padding=4*padding, stride=4)
            
            # fold操作恢复空间维度
            T_lv3 = F.fold(T_lv3_unfold, output_size=dh_img_lv3.size()[-2:],
                           kernel_size=(kernel_size, kernel_size), padding=padding) / overlap_cnt_lv3
            T_lv2 = F.fold(T_lv2_unfold, output_size=(dh_img_lv3.size(2)*2, dh_img_lv3.size(3)*2),
                           kernel_size=(2*kernel_size, 2*kernel_size), padding=2*padding, stride=2) / overlap_cnt_lv2
            T_lv1 = F.fold(T_lv1_unfold, output_size=(dh_img_lv3.size(2)*4, dh_img_lv3.size(3)*4),
                           kernel_size=(4*kernel_size, 4*kernel_size), padding=4*padding, stride=4) / overlap_cnt_lv1
            
            # 最大相关性作为注意力图
            relevance, _ = torch.max(relevance, dim=-1)
            S = relevance.view(relevance.size(0), 1, dh_img_lv3.size(2), dh_img_lv3.size(3))
        
        return S, T_lv3, T_lv2, T_lv1