import os

from torch.utils.data import DataLoader, Dataset

from .dataset import hmDataset
from .transforms import get_train_transform, get_valid_transform

def build_train_loader(cfg, id_list):

    train_transform = get_train_transform(cfg)
    train_dataset = hmDataset(
        ids=id_list,
        data_dir=cfg.DATA_DIR,
        transforms=train_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN_LOADER.BATCH_SIZE,
        num_workers=cfg.TRAIN_LOADER.NUM_WORKERS
    )

    return train_loader

def build_valid_loader(cfg, id_list):

    valid_transform = get_valid_transform(cfg)

    valid_dataset = hmDataset(
        ids=id_list,
        data_dir=cfg.DATA_DIR,
        transforms=valid_transform,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg.VALID_LOADER.BATCH_SIZE,
        num_workers=cfg.VALID_LOADER.NUM_WORKERS
    )

    return valid_loader