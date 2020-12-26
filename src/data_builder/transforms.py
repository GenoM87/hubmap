import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(cfg):

    return A.Compose([
        A.HorizontalFlip(p=cfg.DATASET.H_FLIP_PROB),
        A.Resize(height=cfg.DATASET.IMG_HEIGHT, width=cfg.DATASET.IMG_WIDTH),
    ])

def get_valid_transform(cfg):
    return A.Compose([
        A.Resize(height=cfg.DATASET.IMG_HEIGHT, width=cfg.DATASET.IMG_WIDTH),
    ])

def get_test_transform(cfg):
    return A.Compose([
        A.Resize(height=cfg.DATASET.IMG_HEIGHT, width=cfg.DATASET.IMG_WIDTH),
    ])