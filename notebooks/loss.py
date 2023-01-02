import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.smooth = 1.
        return None

    def forward(self, output, label):
        output = F.sigmoid(output)       
        output = output.view(-1)
        label = label.view(-1)
        intersection = (output * label).sum()                            
        dice = 1. - (2. * intersection + self.smooth) / (output.sum() + label.sum() + self.smooth)  
        bce = F.binary_cross_entropy(output, label, reduction='mean')
        loss = bce + dice
        return loss

    
class Metric(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.smooth = 1e-5
        return None

    def forward(self, output, label):
        output = F.sigmoid(output)       
        output = output.view(-1)
        label = label.view(-1)
        intersection = (output * label).sum()                            
        dice = (2. * intersection + self.smooth) / (output.sum() + label.sum() + self.smooth)
        return dice

