import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveLpPool2d(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        # 可学习的Lp参数，初始化为p=2
        self.p = nn.Parameter(torch.ones(1) * p)

    def forward(self, x):
        # 限制p在合理范围内（1.5-4），避免过极端的pooling
        p = torch.clamp(self.p, 1.5, 4.0)
        
        # 改进：使用更稳定的Lp pooling计算
        # 保留空间结构的同时计算Lp norm
        # 使用dim=[2,3]在H,W维度上做adaptive pooling
        result = (x.abs().pow(p).mean(dim=[2, 3]) + 1e-8).pow(1.0 / p)
        
        return result


class IAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 改进：初始化为接近1.0，使得模块在早期训练中主要采用GMP通路
        # 这样可以更好地保留判别性特征
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels)
        )
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        b, c, h, w = x.size()

        gap = F.adaptive_avg_pool2d(x, 1).view(b, c)
        gmp = F.adaptive_max_pool2d(x, 1).view(b, c)

        alpha_sigmoid = torch.sigmoid(self.alpha)
        w = alpha_sigmoid * gmp + (1 - alpha_sigmoid) * gap
        w = self.mlp(w)
        w = self.bn(w)
        
        # 添加residual连接可选项
        w = w.view(b, c, 1, 1)
        
        # 使用soft thresholding而不是harsh sigmoid
        attention = torch.tanh(w)  # 使用tanh而不是sigmoid，范围[-1,1]更温和
        
        return x * (1 + 0.5 * attention)  # 添加残差项：1 + 0.5*attention


class GhostBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)