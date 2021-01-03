import os

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import albumentations as A
import pandas as pd

from .dataset import hmTrainDataset, hmTestDataset
from .transforms import get_train_transform, get_valid_transform

def build_train_loader(cfg):

    df = pd.read_csv(
        os.path.join(cfg.DATASET.TILE_DIR, 'coord.csv')
    )

    train_ids = set(df['image_id']) - set(cfg.DATASET.VALID_ID)
    train_transform = get_train_transform(cfg)
    train_dataset = hmTrainDataset(
        df = df, 
        img_ids = list(train_ids), 
        cfg = cfg, 
        transforms=train_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=RandomSampler(train_dataset),
        drop_last=True,
        batch_size=cfg.TRAIN_LOADER.BATCH_SIZE,
        num_workers=cfg.TRAIN_LOADER.NUM_WORKERS
    )

    return train_loader

def build_valid_loader(cfg):

    valid_transform = get_valid_transform(cfg)
    df = pd.read_csv(
        os.path.join(cfg.DATASET.TILE_DIR, 'coord.csv')
    )

    valid_dataset = hmTrainDataset(
        df = df, 
        img_ids = cfg.DATASET.VALID_ID, 
        cfg = cfg, 
        transforms=valid_transform
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        drop_last=False,
        batch_size=cfg.VALID_LOADER.BATCH_SIZE,
        num_workers=cfg.VALID_LOADER.NUM_WORKERS
    )

    return valid_loader