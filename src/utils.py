import math
import numpy as np
import logging
import cv2
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


def calc_psnr(img1, img2):
    """计算标准PSNR值"""
    # 假设图像范围是0-255
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # 避免除以零
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_psnr_and_ssim(sr, hr):
    """计算PSNR值，处理不同的输入张量形状"""
    # 准备数据
    sr = (sr+1.) * 127.5  # 从[-1,1]转换到[0,255]
    hr = (hr+1.) * 127.5
    if (sr.size() != hr.size()):
        h_min = min(sr.size(2), hr.size(2))
        w_min = min(sr.size(3), hr.size(3))
        sr = sr[:, :, :h_min, :w_min]
        hr = hr[:, :, :h_min, :w_min]

    # 转换为numpy并确保形状正确
    sr_np = sr.detach().round().cpu().numpy()
    hr_np = hr.detach().round().cpu().numpy()
    
    # 处理批次大小
    if sr_np.shape[0] == 1:
        sr_np = sr_np[0]
        hr_np = hr_np[0]
    else:
        # 计算批次中每个图像的PSNR并取平均
        batch_psnr = 0
        for i in range(sr_np.shape[0]):
            img1 = np.transpose(sr_np[i], (1, 2, 0))
            img2 = np.transpose(hr_np[i], (1, 2, 0))
            batch_psnr += calc_psnr(img1, img2)
        return batch_psnr / sr_np.shape[0]
    
    # 单张图像处理
    sr_np = np.transpose(sr_np, (1, 2, 0))
    hr_np = np.transpose(hr_np, (1, 2, 0))
    
    return calc_psnr(sr_np, hr_np)