import os
import cv2

import pandas as pd
import numpy as np
import rasterio

from utils import create_tile, read_tiff, rle2mask, create_tile_v2
from config import _C as cfg

img_scale = cfg.DATASET.IMG_SCALE 
tile_size = cfg.DATASET.TRAIN_TILE_SIZE
tile_avg_step = cfg.DATASET.TRAIN_TILE_AVG_STEP
tile_min_score = cfg.DATASET.TRAIN_TILE_MIN_SCORE 

if __name__ == "__main__":
    
    df_train = pd.read_csv(os.path.join(cfg.DATA_DIR, 'train.csv'))

    path_tile = f'{img_scale}_{tile_min_score}_{tile_size}_{tile_avg_step}_train'
    IMG_OUT = os.path.join(cfg.DATA_DIR, 'train', path_tile)

    os.makedirs(IMG_OUT, exist_ok=True)
    print(path_tile)

    df_coord = pd.DataFrame()
    for cnt, row in df_train.iloc[:2].iterrows():
        
        img_id, encoding = row['id'], row['encoding']
        path_image = os.path.join(cfg.DATA_DIR, 'train', img_id+'.tiff')

        os.makedirs(os.path.join(IMG_OUT, img_id), exist_ok=True)
        
        print(f'CREATING TILE FOR IMAGE {img_id}')

        res = create_tile_v2(
            img_id,
            df_train,
            cfg,
            phase='train'
        )
        
        df_image = pd.DataFrame()
        coord = np.array(res['coord'])
        df_image['cx']=coord[:,0].astype(np.int32)
        df_image['cy']=coord[:,1].astype(np.int32)
        df_image['cv']=coord[:,2]
        df_image['image_id'] = img_id 

        df_coord = df_coord.append(df_image)

        tile_id = []
        for i in range(len(res['coord'])):
            cx, cy, cv = res['coord'][i]
            s = f'x{cx}_y{cy}'
            tile_id.append(s)

            tile_img = res['img_tile'][i]
            tile_mask = res['mask_tile'][i]

            cv2.imwrite(os.path.join(IMG_OUT, img_id, f'{s}.png'), tile_img)
            cv2.imwrite(os.path.join(IMG_OUT, img_id, f'{s}.mask.png'), tile_mask)

    df_coord.to_csv(os.path.join(IMG_OUT, 'coord.csv'), index=False)

