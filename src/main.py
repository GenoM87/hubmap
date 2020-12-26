import pandas as pd
import numpy as np

import sys
import os
import cv2

import torch
from torch import optim
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from lovasz import lovasz_hinge

from data_builder.builder import build_train_loader, build_valid_loader
from config import _C as cfg

metric = smp.utils.losses.DiceLoss()

def symmetric_lovasz(outputs, targets):
    return 0.5*(lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))

class hmapModel(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.model = smp.Unet('resnet34', encoder_weights='imagenet')
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = symmetric_lovasz(y_hat, y)
        dice_m = metric(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_dice_loss', dice_m)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = metric(y_hat, y)
        self.log('valid_dice_loss', loss)

if __name__ == "__main__":
    data = os.listdir(os.path.join(cfg.DATA_DIR, 'train'))
    train_lsit = list(set([row.split("_")[0] for row in data]))
    train_idx = [row for row in data if row.split("_")[0] in train_lsit[:-2]]
    valid_idx = [row for row in data if row.split("_")[0] not in train_lsit[:-2]]

    train_loader = build_train_loader(cfg, train_idx)
    valid_loader = build_valid_loader(cfg, valid_idx)

    metric = smp.utils.losses.DiceLoss()

    model = hmapModel()

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, train_loader, valid_loader)

    