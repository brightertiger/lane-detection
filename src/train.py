import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = 'cuda:0'

def save_checkpoint(
    epoch: int, 
    model: nn.Module, 
    loss: float, 
    metric: float,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    path: str
) -> None:
    """Save model checkpoint."""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    
    results = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metric': metric,
    }
    
    torch.save(results, path)
    logger.info(f"Checkpoint saved to {path}")


def validate(
    model: nn.Module, 
    dataloader: DataLoader, 
    loss_fn: Callable, 
    metric_fn: Callable,
    device: str = 'cuda'
) -> Tuple[float, float]:
    """Evaluate model on validation data."""
    model.eval()
    losses = []
    metrics = []
    
    with torch.no_grad():
        for sample in dataloader:
            image = sample['image'].float().to(device)
            label = sample['label'].float().to(device)
            
            output = model(image)
            loss = loss_fn(output, label)
            
            losses.append(loss.item())
            metric = metric_fn(output, label)
            metrics.append(metric)
    
    mean_loss = np.mean(losses)
    mean_metric = np.mean(metrics)
    
    return mean_loss, mean_metric


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    loss_fn: Callable,
    metric_fn: Callable,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    save_path: str,
    epochs: int,
    device: str = 'cuda',
    patience: int = 10
) -> Dict[str, List[float]]:
    """Train segmentation model."""
    # Setup
    model = model.to(device)
    best_valid_loss = float('inf')
    early_stop_counter = 0
    history = {
        'train_loss': [], 
        'train_metric': [], 
        'valid_loss': [], 
        'valid_metric': []
    }
    
    # Create log file
    log_path = save_path.replace('model.pt', 'log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,train_metric,valid_loss,valid_metric,learning_rate\n')
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        
        train_losses = []
        train_metrics = []
        
        # Progress bar
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        
        # Train one epoch
        for sample in train_loader:
            # Move data to device
            image = sample['image'].float().to(device)
            label = sample['label'].float().to(device)
            
            # Forward pass
            output = model(image)
            loss = loss_fn(output, label)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_losses.append(loss.item())
            metric = metric_fn(output, label)
            train_metrics.append(metric)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{np.mean(train_losses):.4f}',
                'metric': f'{np.mean(train_metrics):.4f}'
            })
        
        pbar.close()
        
        # Calculate average training metrics
        train_loss = np.mean(train_losses)
        train_metric = np.mean(train_metrics)
        
        # Validate
        valid_loss, valid_metric = validate(model, valid_loader, loss_fn, metric_fn, device)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step(valid_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_metric'].append(train_metric)
        history['valid_loss'].append(valid_loss)
        history['valid_metric'].append(valid_metric)
        
        # Log metrics
        elapsed_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}, "
            f"Valid Loss: {valid_loss:.4f}, Valid Metric: {valid_metric:.4f}, "
            f"LR: {current_lr:.6f}, Time: {elapsed_time:.1f}s"
        )
        
        # Write to log file
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_metric:.6f},{valid_loss:.6f},{valid_metric:.6f},{current_lr:.6f}\n")
        
        # Save checkpoint if validation loss improved
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            early_stop_counter = 0
            save_checkpoint(
                epoch=epoch,
                model=model,
                loss=valid_loss,
                metric=valid_metric,
                optimizer=optimizer,
                scheduler=scheduler,
                path=save_path
            )
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return history

