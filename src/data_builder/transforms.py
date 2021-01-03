import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(cfg):

    return A.Compose([
        #A.OneOf([
        #  A.OpticalDistortion(p=cfg.DATASET.P_OPTICAL_DIST),
        #  A.GridDistortion(p=cfg.DATASET.P_GRID_DIST),
        #  A.IAAPiecewiseAffine(p=cfg.DATASET.P_PIECEWISE_AFFINE),
        #], p=0.3),
        A.OneOf([
          A.HueSaturationValue(10,15,10, p=cfg.DATASET.P_HUE_SATURATION),
          A.CLAHE(clip_limit=2, p=cfg.DATASET.P_CLAHE),
          #A.RandomBrightnessContrast(p=cfg.DATASET.P_RANDOM_BRIGHTNESS),            
        ], p=0.3),
        A.HorizontalFlip(cfg.DATASET.P_HORIZONATL_FLIP),
        A.VerticalFlip(cfg.DATASET.P_VERTICAL_FLIP),
        A.RandomRotate90(cfg.DATASET.P_RANDOM_ROTATE),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=cfg.DATASET.P_SHIFT_SCALE, 
            border_mode=cv2.BORDER_REFLECT
        ),
        A.Resize(height=cfg.DATASET.IMG_HEIGHT, width=cfg.DATASET.IMG_WIDTH),
        A.Normalize(),
        ToTensorV2()
    ])

def get_valid_transform(cfg):
    return A.Compose([
        #A.Resize(height=cfg.DATASET.IMG_HEIGHT, width=cfg.DATASET.IMG_WIDTH),
        A.Normalize(),
        ToTensorV2()
    ])

def get_test_transform(cfg):
    return A.Compose([
        A.Resize(height=cfg.DATASET.IMG_HEIGHT, width=cfg.DATASET.IMG_WIDTH),
        A.Normalize(),
        ToTensorV2()
    ])