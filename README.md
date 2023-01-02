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


├──  source                      # SOURCE CODE
|    ├── data.py                    # Data Loaders  
|    ├── loss.py                    # Loss and Metric Functions
|    ├── model.py                   # Model
|    └── train.py                   # Training Loop


├── notebooks                    # JUPYTER NOTEBOOKS
│   ├── 01-data.ipynb                # Data Preprocessing
│   ├── 02-transform.ipynb           # Data Augmentation
│   ├── 03-model.ipynb               # Model Training
│   ├── 04-evaluate.ipynb            # Model Evaluation

└── app.py                       # GRADIO APP

```

The input data files and trained models are saved as [Kaggle Dataset.](https://www.kaggle.com/datasets/brightertiger/bdd-lane-detection). They may be downloaded and placed in the 'data' folder in this repository for reproducing the results.

### Solution Approach

The solution involves training a UNET based segmentation model relying on a Resnet-34 backbone. DICE + BCE is used a sloss functions and evaluation is done using IoU metric. The final model performance and metrics can be seen below. 

![](/docs/performance.png)

The output from scoring the model looks as follows:

![](/docs/output.png)

The detailed PDF report is available [here]().

### Serving

The model can be served via Gradio interface. The code for the same is in app.py file. Below is the screenshot of the demo. 

![](/docs/app.png)

