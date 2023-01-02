import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        params = {}
        params['encoder_name'] = 'resnet34'
        params['encoder_weights'] = 'imagenet'
        params['in_channels'] = 3
        params['classes'] = 1
        params['activation'] = 'identity'
        self.model = smp.Unet(**params)
        return None
    
    def forward(self, image):
        output = self.model(image)
        output = output.squeeze()
        return output

