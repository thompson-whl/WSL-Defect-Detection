import torch
import torch.nn as nn
from .modules import *

class WSLNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ===== 优化：平衡模型深度和速度 =====
        # 原来：6 层 GhostBlock
        # 优化：4 层 GhostBlock (更快收敛，保留关键特征)
        self.backbone = nn.Sequential(
            GhostBlock(3, 32),
            GhostBlock(32, 64),
            GhostBlock(64, 128),
            GhostBlock(128, 256)
        )

        self.iam = IAM(256)
        self.pool = AdaptiveLpPool2d()

        self.fc = nn.Linear(256, 1)
        self.fc_bn = nn.BatchNorm1d(256)

        # Use a buffer for the center so it moves with the model and is not optimized
        self.register_buffer('center', torch.zeros(256))

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.iam(feat)

        pooled = self.pool(feat)
        pooled = self.fc_bn(pooled)

        logits = self.fc(pooled)

        return logits, pooled, feat