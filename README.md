# BDD Lane Detection

### Overview

This repository contains code to train a Resnet34-backbone-based UNET model for detecting lanes using a small sample (~3k images) from [BDD-Lane-Detection](https://www.bdd100k.com) dataset.

### Code Structure

```
├── data                         # DATA
│   ├── images                       # Sample downloaded from BDD-100k
│   │   ├── train
│   │   └── valid
│   └── labels                       # Masks corresponding to Images 
│       ├── train
│       └── valid


└── source                      # SOURCE CODE
|   ├── data.py                     # Data Loaders  
|    ├── loss.py                    # Loss and Metric Functions
|    ├── model.py                   # Model
|    └── train.py                   # Training Loop


├── notebooks                    # JUPYTER NOTEBOOKS
│   ├── 01-data.ipynb                # Data Preprocessing
│   ├── 02-transform.ipynb           # Data Augmentation
│   ├── 03-model.ipynb               # Model Training
│   ├── 04-evaluate.ipynb            # Model Evaluation

```

The input data files and trained models are saved as [Kaggle Dataset.](https://www.kaggle.com/datasets/brightertiger/bdd-lane-detection)

### Solution Approach

The solution involves training a UNET based segmentation model relying on a Resnet-34 backbone. DICE + BCE is used a sloss functions and evaluation is done using IoU metric. The final model performance and metrics can be seen below. 

![](/images/performance.png)

The output from scoring the model looks as follows:

![](/images/output.png)

