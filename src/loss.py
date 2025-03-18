import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DiceBCELoss(nn.Module):
    """Combined Dice and Binary Cross Entropy loss."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to get probabilities
        output = torch.sigmoid(output)
        
        # Flatten tensors
        output_flat = output.view(-1)
        target_flat = target.view(-1)
        
        # Calculate Dice loss
        intersection = (output_flat * target_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            output_flat.sum() + target_flat.sum() + self.smooth
        )
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy(output_flat, target_flat, reduction='mean')
        
        # Combine losses
        return bce_loss + dice_loss


class IoUMetric(nn.Module):
    """Intersection over Union (IoU) metric for binary segmentation."""
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth
        
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> float:
        # Apply sigmoid and threshold
        output = torch.sigmoid(output)
        output = (output > self.threshold).float()
        target = (target > 0.0).float()
        
        # Flatten tensors
        output_flat = output.view(-1)
        target_flat = target.view(-1)
        
        # Calculate IoU
        intersection = (output_flat * target_flat).sum()
        union = (output_flat + target_flat).gt(0).sum()
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou.item()
 