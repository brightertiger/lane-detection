import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Any, Optional


class SegmentationModel(nn.Module):
    """U-Net based segmentation model."""
    
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None
    ):
        super().__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation or 'identity'
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output.squeeze()
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = 'cuda'):
        """Load model from a checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = cls()
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        except KeyError:
            raise KeyError(f"Invalid checkpoint format. Missing 'model_state_dict'.")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

