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

    def __init__(self, df, img_ids, cfg, transforms=None, preprocessing=None):
        self.img_ids = img_ids
        self.df = df[df['image_id'].isin(img_ids)]
        self.cfg = cfg
        self.tile_dir = cfg.DATASET.TILE_DIR
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx:int):
        row = self.df.iloc[idx]
        path_img = os.path.join(
            self.tile_dir, 
            row['image_id']
        )
        img = cv2.imread(
            os.path.join(path_img, 'x'+str(row['cx'])+'_y'+str(row['cy'])+'.png'),
            cv2.IMREAD_COLOR
        )
        mask = cv2.imread(
            os.path.join(path_img, 'x'+str(row['cx'])+'_y'+str(row['cy'])+'.mask.png'),
            cv2.IMREAD_GRAYSCALE
        )

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask