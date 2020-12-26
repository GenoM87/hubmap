import pathlib
import os
#import torch
from yacs.config import CfgNode as CN

#GENERAL CONFIG

_C = CN()
_C.PROJECT_DIR = str(pathlib.Path(__file__).parent.parent.absolute())
_C.DATA_DIR = os.path.join(_C.PROJECT_DIR, 'data')
#_C.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#Dataset config
_C.DATASET = CN()

#TILE CREATION
_C.DATASET.IMG_SCALE = 0.25
_C.DATASET.TRAIN_TILE_SIZE = 320
_C.DATASET.TRAIN_TILE_AVG_STEP = 160
_C.DATASET.TRAIN_TILE_MIN_SCORE = 0.25
_C.DATASET.TILE_DIR = os.path.join(
    _C.DATA_DIR, 
    'train', 
    f'{_C.DATASET.IMG_SCALE}_{_C.DATASET.TRAIN_TILE_MIN_SCORE}_{_C.DATASET.TRAIN_TILE_SIZE}_{_C.DATASET.TRAIN_TILE_AVG_STEP}_train'
)

 #DATASET AUGMENTATION
_C.DATASET.IMG_HEIGHT = 256
_C.DATASET.IMG_WIDTH = 256
_C.DATASET.H_FLIP_PROB = 0.5
_C.DATASET.NUM_WORKERS = 4

#Loader config
_C.TRAIN_LOADER = CN()
_C.TRAIN_LOADER.BATCH_SIZE = 32
_C.TRAIN_LOADER.NUM_WORKERS = 4

_C.VALID_LOADER = CN()
_C.VALID_LOADER.BATCH_SIZE = 128
_C.VALID_LOADER.NUM_WORKERS = 4

#train config
_C.TRAIN = CN()
_C.TRAIN.LR = 1e-03

#Model config
_C.MODEL = CN()
_C.MODEL.NAME = 'resnet18'
_C.MODEL.PRETRAINING = 'imagenet'