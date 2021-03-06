import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BCEDiceLoss', 'RMSELoss']

# Segmentation Component Loss Function
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

# Volume Component Loss Function
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,edv_output, esv_output,target):
        target[0] = torch.as_tensor(target[0])
        target[0] = target[0].cuda()
        target[1] = torch.as_tensor(target[1])
        target[1] = target[1].cuda()
        return torch.sqrt(self.mse(edv_output, target[0])) + torch.sqrt(self.mse(esv_output, target[1]))