import os

from torch.utils.data import DataLoader, Dataset
import albumentations as A

from .dataset import hmDataset
from .transforms import get_train_transform, get_valid_transform
from segmentation_models_pytorch.encoders import get_preprocessing_fn

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)

def build_train_loader(cfg, id_list):

    train_transform = get_train_transform(cfg)
    train_dataset = hmDataset(
        ids=id_list,
        data_dir=cfg.DATA_DIR,
        transforms=train_transform,
        preprocessing=get_preprocessing(
            get_preprocessing_fn(cfg.MODEL.NAME, pretrained=cfg.MODEL.PRETRAINING)
        )
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
        preprocessing=get_preprocessing(
            get_preprocessing_fn(cfg.MODEL.NAME, pretrained=cfg.MODEL.PRETRAINING)
        )
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg.VALID_LOADER.BATCH_SIZE,
        num_workers=cfg.VALID_LOADER.NUM_WORKERS
    )

    return valid_loader