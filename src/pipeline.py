"""Pipeline for lane detection segmentation model training and inference."""

import os
import yaml
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from tqdm import tqdm
import cv2
import albumentations as A
from torch.utils.data import DataLoader

# Import custom modules
from .data import create_dataloaders, BaseDataset
from .model import SegmentationModel
from .loss import DiceBCELoss, IoUMetric
from .train import train, validate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SegmentationPipeline:
    """Pipeline for training and inference of lane segmentation models."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration from YAML file."""
        self.config = self._load_config(config_path)
        self.device = self._get_device()
        self.model = None
        self.loss_fn = None
        self.metric_fn = None
        self.optimizer = None
        self.scheduler = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def _get_device(self) -> str:
        """Get the appropriate device based on configuration and availability."""
        device = self.config['training'].get('device', 'cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA is not available. Using CPU instead.")
            device = 'cpu'
        return device
    
    def setup(self) -> None:
        """Set up the model, loss function, metric, and optimizer."""
        # Create model
        model_config = self.config['model']
        self.model = SegmentationModel(
            encoder_name=model_config.get('encoder_name', 'resnet34'),
            encoder_weights=model_config.get('encoder_weights', 'imagenet'),
            in_channels=model_config.get('in_channels', 3),
            classes=model_config.get('classes', 1),
            activation=model_config.get('activation')
        )
        self.model = self.model.to(self.device)
        
        # Create loss function and metric
        self.loss_fn = DiceBCELoss()
        self.metric_fn = IoUMetric()
        
        # Create optimizer
        training_config = self.config['training']
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 1e-4),
            weight_decay=training_config.get('weight_decay', 1e-4)
        )
        
        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        logger.info("Model, loss function, and optimizer have been set up.")
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load training and validation data."""
        data_config = self.config['data']
        data_path = data_config.get('path', './data')
        
        # Check if train/valid CSVs exist
        train_csv_path = Path(data_path) / 'train.csv'
        valid_csv_path = Path(data_path) / 'valid.csv'
        
        if train_csv_path.exists() and valid_csv_path.exists():
            logger.info("Loading data from predefined train/valid splits.")
            train_df = pd.read_csv(train_csv_path)
            valid_df = pd.read_csv(valid_csv_path)
        else:
            logger.info("No predefined splits found. Creating splits from available data.")
            # List all image files
            images_path = Path(data_path) / 'images' / 'train'
            if not images_path.exists():
                raise FileNotFoundError(f"Images directory not found: {images_path}")
            
            # Get all jpg files
            all_images = [f.name for f in images_path.glob('*.jpg')]
            if not all_images:
                raise ValueError(f"No images found in {images_path}")
            
            # Create dataframe
            all_df = pd.DataFrame({'image': all_images})
            
            # Split into train/validation
            val_split = data_config.get('val_split', 0.2)
            val_size = int(len(all_df) * val_split)
            train_df = all_df.iloc[:-val_size].reset_index(drop=True)
            valid_df = all_df.iloc[-val_size:].reset_index(drop=True)
            
            # Save splits for reproducibility
            train_df.to_csv(train_csv_path, index=False)
            valid_df.to_csv(valid_csv_path, index=False)
        
        # Create data loaders
        train_loader, valid_loader = create_dataloaders(
            path=data_path,
            train_df=train_df,
            valid_df=valid_df,
            batch_size=data_config.get('batch_size', 8),
            num_workers=data_config.get('num_workers', 4),
            img_size=data_config.get('img_size', 720)
        )
        
        return train_loader, valid_loader
    
    def train(self) -> Dict[str, List[float]]:
        """Train the model."""
        if self.model is None:
            self.setup()
        
        train_loader, valid_loader = self.load_data()
        training_config = self.config['training']
        
        # Create checkpoint directory
        checkpoint_dir = Path(training_config.get('checkpoint_dir', './checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = checkpoint_dir / 'model.pt'
        
        # Train the model
        history = train(
            model=self.model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            save_path=str(save_path),
            epochs=training_config.get('epochs', 50),
            device=self.device,
            patience=training_config.get('patience', 10)
        )
        
        logger.info(f"Training completed. Best model saved to {save_path}")
        return history
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        """Load a trained model from a checkpoint."""
        if checkpoint_path is None:
            # Use default path from config
            checkpoint_dir = self.config['training'].get('checkpoint_dir', './checkpoints')
            checkpoint_path = Path(checkpoint_dir) / 'model.pt'
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        if self.model is None:
            self.setup()
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded from {checkpoint_path}")
    
    def predict(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Predict lane segmentation for a single image."""
        if self.model is None:
            raise ValueError("Model is not loaded. Call setup() or load_checkpoint() first.")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        transform = A.Compose([
            A.PadIfNeeded(
                min_height=self.config['data'].get('pad_size', 736),
                min_width=self.config['data'].get('pad_size', 736),
                value=255
            )
        ])
        
        # Apply transformation
        transformed = transform(image=image)
        image_padded = transformed['image']
        
        # Convert to tensor
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        image_tensor = torch.from_numpy(image_padded.transpose(2, 0, 1)).float() / 255.0
        image_tensor = image_tensor.sub_(torch.tensor(mean).view(3, 1, 1)).div_(torch.tensor(std).view(3, 1, 1))
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(image_tensor).squeeze().cpu().numpy()
        
        # Apply sigmoid
        prediction = 1 / (1 + np.exp(-prediction))
        
        return image_padded, prediction
    
    def predict_batch(self, image_dir: str, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Predict lane segmentation for all images in a directory."""
        if self.model is None:
            raise ValueError("Model is not loaded. Call setup() or load_checkpoint() first.")
        
        # Setup output directory
        if output_dir is None:
            output_dir = self.config['inference'].get('save_dir', './predictions')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
        
        results = []
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Predict
                image, mask = self.predict(str(image_path))
                
                # Create results dict
                result = {
                    'image_path': str(image_path),
                    'image': image,
                    'mask': mask
                }
                results.append(result)
                
                # Save results if requested
                if output_dir:
                    # Create binary mask
                    threshold = self.config['inference'].get('threshold', 0.5)
                    binary_mask = (mask > threshold).astype(np.uint8) * 255
                    
                    # Save mask
                    mask_path = Path(output_dir) / f"{image_path.stem}_mask.png"
                    cv2.imwrite(str(mask_path), binary_mask)
                    
                    # Create and save overlay
                    alpha = self.config['inference'].get('overlay_alpha', 0.5)
                    color = self.config['inference'].get('overlay_color', [0, 255, 0])
                    
                    overlay = image.copy()
                    for c in range(3):
                        overlay[:, :, c] = np.where(
                            binary_mask == 255,
                            image[:, :, c] * (1 - alpha) + color[c] * alpha,
                            image[:, :, c]
                        )
                    
                    overlay_path = Path(output_dir) / f"{image_path.stem}_overlay.png"
                    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
        
        return results

    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate model on validation data."""
        if self.model is None:
            raise ValueError("Model is not loaded. Call setup() or load_checkpoint() first.")
        
        if data_loader is None:
            _, data_loader = self.load_data()
        
        # Evaluate
        loss, metric = validate(
            model=self.model,
            dataloader=data_loader,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            device=self.device
        )
        
        results = {
            'loss': loss,
            'iou': metric
        }
        
        logger.info(f"Evaluation results: Loss={loss:.4f}, IoU={metric:.4f}")
        return results 