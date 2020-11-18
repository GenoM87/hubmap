import os
import datetime

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

class hmDataset(Dataset):

    def __init__(self, ids, data_dir, transforms=None, preprocessing=None):
        self.ids = ids
        self.data_dir = data_dir
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx:int):
        name = self.ids[idx].split('.')[0]
        img = cv2.imread(os.path.join(self.data_dir, 'image', name+'.png'))
        mask = np.load(os.path.join(self.data_dir, 'mask', name+'.npy'))

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask
    
    def show_batch(self):

        id_sample = np.random.choice(range(len(self.ids)), 16)
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))

        for idx, ax in enumerate(axes.ravel()):
            img, mask = self.__getitem__(id_sample[idx])
            ax.imshow(img.permute(1, 2, 0))
            ax.matshow(mask.squeeze(), alpha=0.5)