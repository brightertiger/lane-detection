import cv2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import gradio as gr
import albumentations as A 
from torchvision import transforms
from source.model import Model 

weights = torch.load('./data/model.pt', map_location='cpu')
weights = weights['model_state_dict']
MODEL = Model()
MODEL.load_state_dict(weights)
MODEL = MODEL.eval()

def process(image):
    transform = []
    transform.append(A.PadIfNeeded(min_height=736, min_width=736, value=255))
    transform = A.Compose(transform)
    output = transform(image=image)
    image = output['image']
    orig = image.copy()
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = []
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(transform)
    image = transform(image)
    return orig, image

def score(image):
	global MODEL
	orig, image = process(image)
	pred = MODEL(image.unsqueeze(0)).squeeze().data.numpy()
	pred =  1/(1 + np.exp(-pred))
	return pred

if __name__ == '__main__':
	demo = gr.Interface(score, gr.Image(shape=(720,720)), "image", title="BDD Lane Detection Demo")
	demo.launch()
