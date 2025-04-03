import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import MeanShift

class LTEExtractor(nn.Module):
    def __init__(self, requires_grad=True, rgb_range=1):
        super(LTEExtractor, self).__init__()
        
        # 使用预训练的VGG19初始化
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad
        
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, x):
        x = self.sub_mean(x)
        x_lv1 = self.slice1(x)
        x_lv2 = self.slice2(x_lv1)
        x_lv3 = self.slice3(x_lv2)
        
        # 为了保持与现有代码兼容，返回Q, K, V
        # 这里我们使用不同层级的特征作为Q, K, V
        Q = x_lv1  # 低级特征
        K = x_lv2  # 中级特征
        V = x_lv3  # 高级特征
        
        return Q, K, V