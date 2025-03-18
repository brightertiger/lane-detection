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
from src.data import create_dataloaders, BaseDataset
from src.model import SegmentationModel
from src.loss import DiceBCELoss, IoUMetric
from src.train import train, validate

# ... rest of the file remains unchanged 