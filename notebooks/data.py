import torch
import random
import math
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensor
from torchvision import transforms

class TrainDataset(Dataset):

    def __init__(self, path, data):
        self.path = path
        data = data.reset_index(drop=True)
        self.image = data['image'].tolist()
        self.label = data['image'].map(lambda x : x.replace(".jpg",".png")).tolist()
        transform = []
        transform.append(A.RandomCrop(width=720, height=720))
        transform.append(A.HorizontalFlip(p=0.5))
        transform.append(A.PadIfNeeded(min_height=736, min_width=736, value=255))
        self.transform = A.Compose(transform)
        return None
    
    def __len__(self):
        return len(self.image)

    def __image__(self, index):  
        image = self.path + '/images/train/' + self.image[index]
        label = self.path + '/labels/train/' + self.label[index]
        print(image)
        print(label)
        image = cv2.imread(image)
        label = cv2.imread(label)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = self.transform(image=image, mask=label)
        image, label = output['image'], output['mask']
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = []
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean, std))
        transform = transforms.Compose(transform)
        image = transform(image)
        label = label.min(axis=-1)
        label = (np.array(label) != 255).astype(float)
        label = torch.from_numpy(label).squeeze()
        return image, label

    def __getitem__(self, idx):
        image, label = self.__image__(idx)
        sample = {'idx': self.image[idx].replace(".jpg",""), 'image': image, 'label': label}
        return sample
    
class ValidDataset(Dataset):

    def __init__(self, path, data):
        self.path = path
        data = data.reset_index(drop=True)
        self.image = data['image'].tolist()
        self.label = data['image'].map(lambda x : x.replace(".jpg",".png")).tolist()
        transform = []
        transform.append(A.RandomCrop(width=720, height=720))
        transform.append(A.HorizontalFlip(p=0.5))
        transform.append(A.PadIfNeeded(min_height=736, min_width=736, value=255))
        self.transform = A.Compose(transform)
        return None
    
    def __len__(self):
        return len(self.image)

    def __image__(self, index):  
        image = self.path + '/images/valid/' + self.image[index]
        label = self.path + '/labels/valid/' + self.label[index]
        image = cv2.imread(image)
        label = cv2.imread(label)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = self.transform(image=image, mask=label)
        image, label = output['image'], output['mask']
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = []
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean, std))
        transform = transforms.Compose(transform)
        image = transform(image)
        label = label.min(axis=-1)
        label = (np.array(label) != 255).astype(float)
        label = torch.from_numpy(label).squeeze()
        return image, label

    def __getitem__(self, idx):
        image, label = self.__image__(idx)
        sample = {'idx': self.image[idx].replace(".jpg",""), 'image': image, 'label': label}
        return sample
    
def dataLoader(path, train, valid):
    train = TrainDataset(path, train)
    valid = ValidDataset(path, valid)
    print('Train Images:', len(train), 'Valid Images:', len(valid))
    return train, valid