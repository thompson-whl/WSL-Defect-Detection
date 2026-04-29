import torch
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


def hypersphere_loss(features, labels, center):
    center = center.to(features.device).to(features.dtype)
    labels = labels.view(-1)
    dist = torch.norm(features - center, dim=1) 

    normal = labels == 0
    anomaly = labels == 1

    loss = torch.tensor(0.0, device=features.device, dtype=features.dtype)
    if normal.sum() > 0:
        loss += dist[normal].mean()

    if anomaly.sum() > 0:
        loss += torch.clamp(1 - dist[anomaly], min=0).mean()

    return loss