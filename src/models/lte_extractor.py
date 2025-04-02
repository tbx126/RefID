# src/models/lte_extractor.py

import torch.nn as nn

class LTEExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.Q_head = nn.Conv2d(256, 256, kernel_size=1)
        self.K_head = nn.Conv2d(256, 256, kernel_size=1)
        self.V_head = nn.Conv2d(256, 256, kernel_size=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return self.Q_head(x), self.K_head(x), self.V_head(x)