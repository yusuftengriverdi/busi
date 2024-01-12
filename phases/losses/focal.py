# To handle class imbalance towards normal and malignant.
import torch

import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.3, gamma=1, weight= None, reduction='sum'):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        
        elif self.reduction == 'sum':
            return focal_loss.sum()
        
        else:
            return focal_loss