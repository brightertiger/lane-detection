import torch
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, Tuple, List, Optional, Union, Any

class BaseDataset(Dataset):
    """Base dataset class for image segmentation tasks."""
    
    def __init__(
        self, 
        path: str, 
        data: pd.DataFrame, 
        split: str,
        img_size: int = 720,
        pad_size: int = 736,
        pad_value: int = 255,
        apply_augmentation: bool = True
    ):
        self.path = path
        self.split = split
        data = data.reset_index(drop=True)
        self.image_paths = data['image'].tolist()
        self.label_paths = data['image'].map(lambda x: x.replace(".jpg", ".png")).tolist()
        
        # Create transformation pipeline
        transform = []
        if apply_augmentation:
            transform.append(A.RandomCrop(width=img_size, height=img_size))
            transform.append(A.HorizontalFlip(p=0.5))
        else:
            transform.append(A.CenterCrop(width=img_size, height=img_size))
            
        transform.append(A.PadIfNeeded(min_height=pad_size, min_width=pad_size, value=pad_value))
        self.transform = A.Compose(transform)
        
        # Normalization values (ImageNet)
        self.mean = [0.485, 0.456, 0.406] 
        self.std = [0.229, 0.224, 0.225]
    
    def __len__(self) -> int:
        return len(self.image_paths)

    def load_image(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  
        """Load and preprocess an image and its label mask."""
        image_path = f"{self.path}/images/{self.split}/{self.image_paths[index]}"
        label_path = f"{self.path}/labels/{self.split}/{self.label_paths[index]}"
        
        # Load images
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        output = self.transform(image=image, mask=label)
        image, label = output['image'], output['mask']
        
        # Normalize and convert image to tensor
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        image = image_transform(image)
        
        # Process label
        label = label.min(axis=-1)
        label = (np.array(label) != 255).astype(float)
        label = torch.from_numpy(label).squeeze()
        
        return image, label

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, label = self.load_image(idx)
        sample = {
            'idx': self.image_paths[idx].replace(".jpg", ""), 
            'image': image, 
            'label': label
        }
        return sample


class TrainDataset(BaseDataset):
    """Dataset for training data."""
    
    def __init__(
        self, 
        path: str, 
        data: pd.DataFrame,
        img_size: int = 720,
        pad_size: int = 736
    ):
        super().__init__(
            path=path, 
            data=data, 
            split='train',
            img_size=img_size,
            pad_size=pad_size,
            apply_augmentation=True
        )


class ValidDataset(BaseDataset):
    """Dataset for validation data."""
    
    def __init__(
        self, 
        path: str, 
        data: pd.DataFrame,
        img_size: int = 720,
        pad_size: int = 736
    ):
        super().__init__(
            path=path, 
            data=data, 
            split='valid',
            img_size=img_size,
            pad_size=pad_size,
            apply_augmentation=False
        )


def create_dataloaders(
    path: str, 
    train_df: pd.DataFrame, 
    valid_df: pd.DataFrame,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 720
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoader objects for training and validation."""
    train_dataset = TrainDataset(path, train_df, img_size=img_size)
    valid_dataset = ValidDataset(path, valid_df, img_size=img_size)
    
    print(f'Train Images: {len(train_dataset)}, Valid Images: {len(valid_dataset)}')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader