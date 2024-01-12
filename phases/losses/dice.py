# To handle class imbalance towards normal and malignant.
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchmetrics.classification import Dice

class DiceLoss(nn.Module):
    def __init__(self, threshold=0.5, reduction='sum', return_score = False):
        super(DiceLoss, self).__init__()

        self.threshold = threshold
        self.reduction = reduction
        self.return_score = return_score

    def _compute_dice_coef(self, input, target):
        return 2 * (input * target).sum() / ((input**2).sum() + (target**2).sum())
        
    def _binarize_pred(self, input):
        return (input > self.threshold).int()

    def forward(self, input, target):
        
        dice_score = self._compute_dice_coef(self._binarize_pred(input), target)
        dice_loss = Variable((1.0-dice_score), requires_grad=True)

        if self.return_score:
            return dice_score
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        
        elif self.reduction == 'sum':
            return dice_loss.sum()
        
        else:
            return dice_loss
