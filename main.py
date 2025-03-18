"""Command-line interface for lane detection segmentation model pipeline."""

import argparse
import logging
from pathlib import Path
import yaml
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from src.pipeline import SegmentationPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Lane Detection Segmentation Pipeline")
    
    # Main operation mode
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'evaluate', 'predict'],
                        help="Operation mode: train, evaluate, or predict")
    
    # Common arguments
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    
    # Training arguments
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    
    # Prediction arguments
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint for prediction/evaluation")
    parser.add_argument("--input", type=str, help="Input image or directory for prediction")
    parser.add_argument("--output", type=str, help="Output directory for prediction results")
    
    # Evaluation arguments
    parser.add_argument("--split", type=str, default="valid", choices=['train', 'valid', 'test'],
                        help="Data split to evaluate on")
    
    return parser.parse_args()


def visualize_prediction(image, mask, threshold=0.5):
    """Visualize prediction results."""
    # Create binary mask
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Create overlay
    overlay = image.copy()
    overlay[binary_mask == 1, 0] = 0  # Remove red channel where mask is present
    overlay[binary_mask == 1, 1] = 255  # Set green channel where mask is present
    overlay[binary_mask == 1, 2] = 0  # Remove blue channel where mask is present
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(mask, cmap="jet")
    axes[1].set_title("Probability Map")
    axes[1].axis("off")
    
    axes[2].imshow(overlay)
    axes[2].set_title("Segmentation Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create pipeline with provided config
    pipeline = SegmentationPipeline(args.config)
    
    if args.mode == 'train':
        # Training mode
        pipeline.setup()
        if args.resume:
            logger.info(f"Resuming training from checkpoint: {args.resume}")
            pipeline.load_checkpoint(args.resume)
        
        history = pipeline.train()
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['valid_loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_metric'], label='Train')
        plt.plot(history['valid_metric'], label='Validation')
        plt.title('IoU Metric')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        logger.info("Training history saved to training_history.png")
    
    elif args.mode == 'evaluate':
        # Evaluation mode
        pipeline.setup()
        
        # Load checkpoint
        checkpoint_path = args.checkpoint
        if checkpoint_path:
            pipeline.load_checkpoint(checkpoint_path)
        else:
            # Try to load from default location
            logger.info("No checkpoint specified, trying to load from default location")
            try:
                pipeline.load_checkpoint()
            except FileNotFoundError:
                logger.error("No checkpoint found. Please specify a checkpoint with --checkpoint")
                return
        
        # Evaluate
        results = pipeline.evaluate()
        logger.info(f"Evaluation completed: Loss={results['loss']:.4f}, IoU={results['iou']:.4f}")
    
    elif args.mode == 'predict':
        # Prediction mode
        if not args.input:
            logger.error("Input image or directory must be specified with --input")
            return
        
        pipeline.setup()
        
        # Load checkpoint
        checkpoint_path = args.checkpoint
        if checkpoint_path:
            pipeline.load_checkpoint(checkpoint_path)
        else:
            # Try to load from default location
            logger.info("No checkpoint specified, trying to load from default location")
            try:
                pipeline.load_checkpoint()
            except FileNotFoundError:
                logger.error("No checkpoint found. Please specify a checkpoint with --checkpoint")
                return
        
        # Check if input is file or directory
        input_path = Path(args.input)
        if input_path.is_file():
            # Single file prediction
            logger.info(f"Predicting for single image: {input_path}")
            image, mask = pipeline.predict(str(input_path))
            
            # Save or display results
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save mask
                mask_path = output_dir / f"{input_path.stem}_mask.png"
                cv2.imwrite(str(mask_path), (mask > 0.5).astype(np.uint8) * 255)
                
                # Save overlay
                overlay = image.copy()
                binary_mask = (mask > 0.5).astype(np.bool)
                overlay[binary_mask, 1] = 255  # Green channel
                overlay_path = output_dir / f"{input_path.stem}_overlay.png"
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                
                logger.info(f"Results saved to {output_dir}")
            else:
                # Display if not saving
                visualize_prediction(image, mask)
        
        elif input_path.is_dir():
            # Batch prediction
            logger.info(f"Predicting for images in directory: {input_path}")
            results = pipeline.predict_batch(str(input_path), args.output)
            logger.info(f"Processed {len(results)} images")
        
        else:
            logger.error(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main() 