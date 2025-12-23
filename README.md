# BDD Lane Detection

### Overview

This repository contains code to train a Resnet34-backbone-based U-Net model for detecting lanes using a small sample (~3k images) from [BDD-Lane-Detection](https://www.bdd100k.com) dataset. The codebase has been refactored and improved for better maintainability, performance, and extensibility.

### Code Structure

```
├── config.yaml               # Configuration file for all parameters
├── main.py                   # Command-line interface
├── app.py                    # Gradio web interface
├── requirements.txt          # Python dependencies
├── data                      # DATA
│   ├── images                    # Sample downloaded from BDD-100k
│   │   ├── train
│   │   └── valid
│   ├── labels                    # Masks corresponding to Images 
│   │   ├── train
│   │   └── valid
│   ├── train.csv                 # Training split
│   ├── valid.csv                 # Validation split
│   └── model.pt                  # Pretrained model checkpoint

├── src                       # SOURCE CODE
│   ├── data.py                  # Data loaders with improved dataset classes
│   ├── loss.py                  # Loss and metric functions (DiceBCE and IoU)
│   ├── model.py                 # Improved model with class methods
│   ├── train.py                 # Enhanced training pipeline
│   ├── pipeline.py              # End-to-end pipeline for training and inference
│   └── config.py                # Configuration utilities

├── source                    # BACKWARD COMPATIBILITY ALIASES
│   └── *.py                     # Wrapper modules for notebook compatibility

├── notebooks                 # JUPYTER NOTEBOOKS
│   ├── 01-data.ipynb             # Data Preprocessing
│   ├── 02-transform.ipynb        # Data Augmentation
│   ├── 03-model.ipynb            # Model Training
│   ├── 04-evaluate.ipynb         # Model Evaluation

├── checkpoints               # Saved model checkpoints
├── logs                      # Training logs
└── predictions               # Saved prediction results
```

- The input data files and trained models are saved as [Kaggle Dataset](https://www.kaggle.com/datasets/brightertiger/bdd-lane-detection). They may be downloaded and placed in the 'data' folder in this repository for reproducing the results.

- The python files in the `src` folder (renamed from 'source') contain the implementations of the model, loss, training loop, data loaders, etc., now with improved type hints, documentation, and error handling.

- Jupyter notebooks call the classes and functions implemented in the source files for execution.

### New Features

#### Configuration Management
- YAML-based configuration (`config.yaml`) for easy parameter management
- Organized into data, model, training, augmentation, and inference sections

#### Pipeline Architecture
- End-to-end pipeline for training, evaluation, and inference
- Modular components for better code organization
- Automatic data splitting if predefined splits aren't available

#### Improved Training
- Better checkpointing with optimizer state
- Early stopping to prevent overfitting
- Proper learning rate scheduling
- Enhanced logging and progress tracking

#### Advanced Visualization
- Tools for visualizing predictions and training history
- Overlay visualization of segmentation masks

#### Command-line Interface
- Train, evaluate, and perform inference from the command line
- Flexible arguments for different workflows

|Notebook|Description|Link|
|--------|-----------|-----|
|01-data.ipynb| Contains information on datasets, image sizes and labels|[:link:](notebooks/01-data.ipynb)|
|02-transform.ipynb| Experimentations with augmentations like RandomCrop and Horizontal Flips|[:link:](notebooks/02-transform.ipynb)|
|03-model.ipynb| Trains the UNet Model|[:link:](notebooks/03-model.ipynb)|
|04-evaluate.ipynb| Evaluated the model performance on random images from validation set|[:link:](notebooks/04-evaluate.ipynb)|

### Solution Approach

The solution involves training a U-Net based segmentation model relying on a ResNet-34 backbone. DICE + BCE is used as loss function and evaluation is done using IoU metric. The final model performance and metrics can be seen below. 

![](/docs/performance.png)

The output from scoring the model looks as follows:

![](/docs/output.png)

The detailed PDF report is available [here](report.pdf).

### Usage

#### Configuration

The project now uses a YAML configuration file (`config.yaml`) for managing parameters:

```yaml
# Example configuration
data:
  path: "./data"
  img_size: 720
  batch_size: 8

model:
  encoder_name: "resnet34"
  encoder_weights: "imagenet"

training:
  epochs: 50
  learning_rate: 0.0001
```

#### Training

Train the model using the command-line interface:

```bash
python main.py --mode train --config config.yaml
```

To resume training from a checkpoint:

```bash
python main.py --mode train --config config.yaml --resume checkpoints/model.pt
```

#### Evaluation

Evaluate model performance on the validation set:

```bash
python main.py --mode evaluate --config config.yaml --checkpoint checkpoints/model.pt
```

#### Prediction

Make predictions on a single image:

```bash
python main.py --mode predict --config config.yaml --input test_image.jpg --output predictions/
```

Process a directory of images:

```bash
python main.py --mode predict --config config.yaml --input test_images/ --output predictions/
```

### Serving

#### Web Interface

The model can be served via Gradio interface. The code for the same is in `app.py` file. Below is the screenshot of the demo. It's hosted on [Huggingface Spaces](https://huggingface.co/spaces/brightertiger/bdd-lane-detection).

```bash
python app.py
```

![](/docs/app.png)

The improved Gradio interface now provides three outputs:
1. Original image
2. Lane mask (green overlay)
3. Combined visualization

### Requirements

Major dependencies include:

- torch>=2.0.0
- torchvision>=0.15.0
- numpy>=1.24.0
- pandas>=2.0.0
- opencv-python>=4.8.0
- albumentations>=1.3.0
- segmentation-models-pytorch>=0.3.0
- gradio>=4.0.0
- matplotlib>=3.7.0
- PyYAML>=6.0

See `requirements.txt` for the complete list with version specifications.

Install dependencies:

```bash
pip install -r requirements.txt
```

