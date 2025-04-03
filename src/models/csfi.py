import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


class SFE(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(3, n_feats)

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                     res_scale=res_scale))

        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class CSFI2(nn.Module):
    def __init__(self, n_feats):
        super(CSFI2, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats, 2)
        self.conv21 = conv3x3(n_feats, n_feats)

        self.conv_merge1 = conv3x3(n_feats * 2, n_feats)
        self.conv_merge2 = conv3x3(n_feats * 2, n_feats)

    def forward(self, x1, x2):  # x1: (H, W), x2: (H/2, W/2)
        x12 = F.relu(self.conv12(x1))
        x21 = F.interpolate(x2, scale_factor=2, mode='bicubic')

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12), dim=1)))

        return x1, x2


class CSFI3(nn.Module):
    def __init__(self, n_feats):
        super(CSFI3, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats, 2)
        self.conv13_1 = conv1x1(n_feats, n_feats, 2)
        self.conv13_2 = conv1x1(n_feats, n_feats, 2)

        self.conv21 = conv3x3(n_feats, n_feats)
        self.conv23 = conv1x1(n_feats, n_feats, 2)

        self.conv31 = conv3x3(n_feats, n_feats)
        self.conv32 = conv3x3(n_feats, n_feats)

        self.conv_merge1 = conv3x3(n_feats * 3, n_feats)
        self.conv_merge2 = conv3x3(n_feats * 3, n_feats)
        self.conv_merge3 = conv3x3(n_feats * 3, n_feats)

    def forward(self, x1, x2, x3):  # x1: (H, W), x2: (H/2, W/2), x3: (H/4, W/4)
        x12 = F.relu(self.conv12(x1))
        x13 = F.relu(self.conv13_1(x1))
        x13 = F.relu(self.conv13_2(x13))

        x21 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x2))

        x31 = F.interpolate(x3, scale_factor=4, mode='bicubic')
        x32 = F.interpolate(x3, scale_factor=2, mode='bicubic')

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21, x31), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12, x32), dim=1)))
        x3 = F.relu(self.conv_merge3(torch.cat((x3, x13, x23), dim=1)))

        return x1, x2, x3


class MergeTail(nn.Module):
    def __init__(self, n_feats):
        super(MergeTail, self).__init__()
        self.conv31 = conv1x1(n_feats, n_feats)
        self.conv21 = conv1x1(n_feats, n_feats)
        self.conv_merge = conv3x3(n_feats * 3, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats // 2)
        self.conv_tail2 = conv1x1(n_feats // 2, 3)

    def forward(self, x1, x2, x3):
        x31 = F.interpolate(x3, scale_factor=4, mode='bicubic')
        x31 = F.relu(self.conv31(x31))

        x21 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x21 = F.relu(self.conv21(x21))

        x = F.relu(self.conv_merge(torch.cat((x1, x21, x31), dim=1)))
        x = self.conv_tail1(x)
        x = self.conv_tail2(x)
        x = torch.clamp(x, -1, 1)

        return x


class CSFI(nn.Module):
    """重新实现的CSFI，完全兼容RefGT的FeatureFusion模块"""
    def __init__(self, in_channels=256, n_feats=64, res_depth=[3, 3, 3, 3], res_scale=0.1):
        super(CSFI, self).__init__()
        self.n_feats = n_feats
        self.num_res_blocks = res_depth
        
        # 浅层特征提取
        self.SFE = SFE(self.num_res_blocks[0], n_feats, res_scale)
        
        # 第一阶段
        self.conv11_head = conv3x3(64 + n_feats, n_feats)
        self.RB11 = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))
        self.conv11_tail = conv3x3(n_feats, n_feats)
        
        # 第一层到第二层的下采样
        self.conv12 = conv3x3(n_feats, n_feats, 2)
        
        # 第二阶段
        self.conv22_head = conv3x3(128 + n_feats, n_feats)
        self.ex12 = CSFI2(n_feats)
        self.RB21 = nn.ModuleList()
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB21.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))
            self.RB22.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))
        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)
        
        # 第二层到第三层的下采样
        self.conv23 = conv3x3(n_feats, n_feats, 2)
        
        # 第三阶段
        self.conv33_head = conv3x3(256 + n_feats, n_feats)
        self.ex123 = CSFI3(n_feats)
        self.RB31 = nn.ModuleList()
        self.RB32 = nn.ModuleList()
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB31.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))
            self.RB32.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))
            self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))
        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)
        
        # 合并尾部
        self.merge_tail = MergeTail(n_feats)
    
    def forward(self, x, S=None, T_lv3=None, T_lv2=None, T_lv1=None):
        """
        参数:
            x: 输入图像，范围[-1,1]
            S: 注意力图
            T_lv3, T_lv2, T_lv1: 不同层级的转移特征
        返回:
            去雾图像，范围[-1,1]
        """
        # 浅层特征提取
        x = self.SFE(x)  # (H, W)
        
        # 第一阶段
        x11 = x
        
        # 软注意力处理
        x11_res = x11
        x11_res = torch.cat((x11_res, T_lv1), dim=1)
        x11_res = self.conv11_head(x11_res)
        x11_res = x11_res * F.interpolate(S, scale_factor=4, mode='bicubic')
        x11 = x11 + x11_res
        x11_res = x11
        
        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res
        
        # 第二阶段
        x21 = x11
        x21_res = x21  # (H, W)
        x22 = self.conv12(x11)  # (H/2, W/2)
        
        # 软注意力处理
        x22_res = x22  # (H/2, W/2)
        x22_res = torch.cat((x22_res, T_lv2), dim=1)
        x22_res = self.conv22_head(x22_res)
        x22_res = x22_res * F.interpolate(S, scale_factor=2, mode='bicubic')
        x22 = x22 + x22_res
        x22_res = x22
        
        x21_res, x22_res = self.ex12(x21_res, x22_res)  # (H, W), (H/2, W/2)
        
        for i in range(self.num_res_blocks[2]):
            x21_res = self.RB21[i](x21_res)
            x22_res = self.RB22[i](x22_res)
        
        x21_res = self.conv21_tail(x21_res)
        x22_res = self.conv22_tail(x22_res)
        x21 = x21 + x21_res
        x22 = x22 + x22_res
        
        # 第三阶段
        x31 = x21  # (H, W)
        x31_res = x31
        x32 = x22  # (H/2, W/2)
        x32_res = x32
        x33 = self.conv23(x22)  # (H/4, W/4)
        
        # 软注意力处理
        x33_res = x33
        x33_res = torch.cat((x33_res, T_lv3), dim=1)
        x33_res = self.conv33_head(x33_res)
        x33_res = x33_res * S
        x33 = x33 + x33_res  # (H/4, W/4)
        x33_res = x33
        
        x31_res, x32_res, x33_res = self.ex123(x31_res, x32_res, x33_res)
        
        for i in range(self.num_res_blocks[3]):
            x31_res = self.RB31[i](x31_res)
            x32_res = self.RB32[i](x32_res)
            x33_res = self.RB33[i](x33_res)
        
        x31_res = self.conv31_tail(x31_res)
        x32_res = self.conv32_tail(x32_res)
        x33_res = self.conv33_tail(x33_res)
        x31 = x31 + x31_res
        x32 = x32 + x32_res
        x33 = x33 + x33_res
        
        # 最终合并
        x = self.merge_tail(x31, x32, x33)
        
        return x