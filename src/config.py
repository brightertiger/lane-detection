"""Configuration settings for the segmentation project."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any


@dataclass
class DataConfig:
    data_path: str = "./data"
    img_size: int = 720
    pad_size: int = 736
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class ModelConfig:
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    in_channels: int = 3
    classes: int = 1


@dataclass
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 10
    device: str = "cuda"
    checkpoint_dir: str = "./checkpoints"


@dataclass
class Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create a Config object from a dictionary."""
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config
        ) 