"""Gradio web interface for lane detection using the trained segmentation model."""

import cv2
import torch
import numpy as np
import gradio as gr
import albumentations as A
from torchvision import transforms
from typing import Tuple, Union, List
from pathlib import Path

from src.model import SegmentationModel


def load_model(checkpoint_path: str, device: str = "cpu") -> SegmentationModel:
    """Load a trained model from checkpoint file."""
    model = SegmentationModel.from_pretrained(checkpoint_path, device)
    model.eval()
    return model


def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
    """Preprocess an input image for the model."""
    # Apply padding transformation
    transform = A.Compose([
        A.PadIfNeeded(min_height=736, min_width=736, value=255)
    ])
    transformed = transform(image=image)
    padded_img = transformed['image']
    
    # Keep a copy of the original padded image for visualization
    original_padded = padded_img.copy()
    
    # Apply normalization and convert to tensor
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tensor_img = tensor_transform(padded_img)
    
    return original_padded, tensor_img


def predict_mask(model: SegmentationModel, image: np.ndarray) -> np.ndarray:
    """Generate lane segmentation mask from input image."""
    # Preprocess the image
    original_padded, tensor_img = preprocess_image(image)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(tensor_img.unsqueeze(0)).squeeze().cpu().numpy()
    
    # Apply sigmoid to get probability map
    prediction = 1 / (1 + np.exp(-prediction))
    
    return prediction


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, 
                 color: List[int] = [0, 255, 0]) -> np.ndarray:
    """Overlay segmentation mask on the original image."""
    # Create a colored mask
    colored_mask = np.zeros_like(image)
    for c in range(3):
        colored_mask[:, :, c] = mask * color[c]
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    
    return overlay


def process_and_display(image: np.ndarray, model: SegmentationModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process image and return original, prediction mask, and overlay."""
    # Ensure image is in RGB format (Gradio might provide BGR)
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]
    
    # Get prediction mask
    mask = predict_mask(model, image)
    
    # Create binary mask for visualization
    binary_mask = (mask > 0.5).astype(np.float32)
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 1] = binary_mask * 255  # Green channel
    
    # Create overlay
    overlay = overlay_mask(image, binary_mask)
    
    return image, colored_mask, overlay


# Load the model
MODEL = load_model('./data/model.pt')


def gradio_interface(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradio interface function."""
    return process_and_display(image, MODEL)


if __name__ == '__main__':
    # Create Gradio interface
    demo = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Image(type="numpy", label="Input Image", shape=(720, 720)),
        outputs=[
            gr.Image(type="numpy", label="Original"),
            gr.Image(type="numpy", label="Lane Mask"),
            gr.Image(type="numpy", label="Overlay")
        ],
        title="Lane Detection",
        description="Upload an image to detect lanes using a U-Net segmentation model.",
        examples=[
            ["examples/road1.jpg"],
            ["examples/road2.jpg"]
        ] if Path("examples").exists() else None,
        allow_flagging="never"
    )
    
    # Launch the interface
    demo.launch()
