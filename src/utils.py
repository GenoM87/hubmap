import os

import cv2
import numpy as np
import tifffile as tiff

import rasterio
from rasterio.windows import Window

identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def read_tiff(img_path):
    img = tiff.imread(img_path)
    img = np.squeeze(img)
    if img.shape[0]==3:
        img = img.transpose(1,2,0)
        img = np.ascontiguousarray(img)
    
    return img

def create_tile_v2(img_id, df, cfg, phase='train'):
    '''
    
    '''
    #Definizione parametri per creazone tile (da modificare per img_scale)
    img_scale = cfg.DATASET.IMG_SCALE
    tile_min_score = cfg.DATASET.TRAIN_TILE_MIN_SCORE
    tile_size = cfg.DATASET.TRAIN_TILE_SIZE
    tile_avg_step = cfg.DATASET.TRAIN_TILE_AVG_STEP

    tile_size = tile_size / img_scale
    tile_avg_step = tile_avg_step / img_scale

    half = int(tile_size // 2)

    if phase=='train':
        path_img = os.path.join(cfg.DATA_DIR, 'train', img_id+'.tiff')
    else:
        path_img = os.path.join(cfg.DATA_DIR, 'train', img_id+'.tiff')
        
    dataset = rasterio.open(path_img, transform=identity, num_threads = 'all_cpus')
    h, w = dataset.shape
    
    #creazione della mask
    rle = df.loc[df['id']==img_id]['encoding'].values[0]
    mask = rle2mask(rle, (w, h))

    #creazione delle coordinate
    coord_x = np.linspace(half, w-half, int(np.ceil((w-tile_size)/tile_avg_step)), dtype=int)
    coord_y = np.linspace(half, h-half, int(np.ceil((h-tile_size)/tile_avg_step)), dtype=int)

    coord = []
    reject = []

    tile_img = []
    tile_mask = []
    for cy in coord_y:
        for cx in coord_x:
            #leggo l'immagine dal formato originale
            img = dataset.read(
                [1,2, 3], 
                window=Window.from_slices((cy-half, cy+half), (cx-half, cx+half))
            ).transpose(1,2,0)

            mask_t = mask[cy-half:cy+half, cx-half:cx+half]
            
            #faccio resize per tile scale
            img = cv2.resize(
                img, 
                dsize=None, 
                fx=img_scale, 
                fy=img_scale, 
                interpolation = cv2.INTER_AREA
            )

            mask_t = cv2.resize(
                mask_t, 
                dsize=None, 
                fx=img_scale, 
                fy=img_scale, 
                interpolation = cv2.INTER_AREA
            )
            
            #filtro via per le immagini che non contengono nulla
            structure = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            structure = (structure[:, :, 1] > 32).astype(np.uint8)
            structure = structure.astype(np.float32)
            
            curr_val = structure.mean()
            
            if curr_val>tile_min_score:
                coord.append([int(cx*img_scale), int(cy*img_scale), curr_val])
                tile_img.append(img)
                tile_mask.append(mask_t)
            else:
                reject.append([int(cx*img_scale), int(cy*img_scale), curr_val])
    
    return {
        'coord': coord,
        'reject': reject,
        'img_tile': tile_img,
        'mask_tile': tile_mask
    }



def create_tile(img, mask, img_scale, tile_min_score, tile_size, tile_avg_step):
    '''
    img: tiff img full shape (H,W,C)
    mask: mask come from rle_decode (H,W)
    img_scale: scale to resize original image
    img_min_score: filter image with min score (prevent empty tile)
    tile_size: size of the tile
    tile_avg_step: average step for tile creation
    '''
    
    half = tile_size//2
    #TODO: provare altre interpolazioni #INTER LINEAR ecc..
    img_small = cv2.resize(img, dsize=None, fx=img_scale, fy=img_scale, interpolation = cv2.INTER_AREA)
    mask_small = cv2.resize(mask.astype(np.uint8), dsize=None, fx=img_scale, fy=img_scale, interpolation = cv2.INTER_AREA)
    
    structure = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
    structure = (structure[:, :, 1] > 32).astype(np.uint8)
    
    structure = structure.astype(np.float32)
    
    h, w, _ = img_small.shape
    
    #creo il linspace per "tagliare" l'immagine in tile
    coord_x = np.linspace(half, w-half, int(np.ceil((w-tile_size)/tile_avg_step)), dtype=int)
    coord_y = np.linspace(half, h-half, int(np.ceil((h-tile_size)/tile_avg_step)), dtype=int)
    
    coord = []
    reject = []
    for cy in coord_y:
        for cx in coord_x:
            #filtro via per le immagini che non contengono nulla
            curr_val = structure[cy-half:cy+half, cx-half:cx+half].mean()
            
            if curr_val>tile_min_score:
                coord.append([cx, cy, curr_val])
            else:
                reject.append([cx, cy, curr_val])
    
    tile_mask = []
    tile_img = []
    for cx, cy, cv in coord:
        im = img_small[cy-half:cy+half,cx-half:cx+half]
        ms = mask_small[cy-half:cy+half,cx-half:cx+half]
        
        tile_img.append(im)
        tile_mask.append(ms)
        
    return {
        'coord': coord,
        'reject': reject,
        'img': img_small,
        'img_tile': tile_img,
        'structure': structure,
        'mask_small': mask_small,
        'mask_tile': tile_mask
    }

def create_tile_subm_v2(img_id, img_scale, tile_min_score, tile_size, tile_avg_step):
    '''
    create_tile_v2: funzione per creare i tile usando rasterio come engine. 
        funzione per la prediction
    '''
    tile_size = tile_size / img_scale
    tile_avg_step = tile_avg_step / img_scale

    half = tile_size//2
    
    path_img = os.path.join(DATA, img_id+'.tiff')
    dataset = rasterio.open(path_img, trasform=identity, num_threads = 'all_cpus')
    
    h, w = dataset.shape
    
    coord_x = np.linspace(half, w-half, int(np.ceil((w-tile_size)/tile_avg_step)), dtype=int)
    coord_y = np.linspace(half, h-half, int(np.ceil((h-tile_size)/tile_avg_step)), dtype=int)
    
    coord = []
    reject = []
    tile_img = []
    for cy in coord_y:
        for cx in coord_x:
            #leggo l'immagine dal formato originale
            img = dataset.read(
                [1,2, 3], 
                window=Window.from_slices((cy-half, cy+half), (cx-half, cx+half))
            ).transpose(1,2,0)
            
            #faccio resize per tile scale
            img = cv2.resize(
                img, 
                dsize=None, 
                fx=img_scale, 
                fy=img_scale, 
                interpolation = cv2.INTER_AREA
            )
            
            #filtro via per le immagini che non contengono nulla
            structure = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            structure = (structure[:, :, 1] > 32).astype(np.uint8)
            structure = structure.astype(np.float32)
            
            curr_val = structure.mean()
            
            if curr_val>tile_min_score:
                coord.append([int(cx*img_scale), int(cy*img_scale), curr_val])
                tile_img.append(img)
            else:
                reject.append([int(cx*img_scale), int(cy*img_scale), curr_val])
    
    return {
        'coord': coord,
        'reject': reject,
        'img_tile': tile_img,
    }

def create_tile_subm(img, img_scale, tile_min_score, tile_size, tile_avg_step):
    '''
    create_tile_subm()> Dict: 
        funzione per creare i tile dell'immagine per la submission (non è presente la mask)

    INPUT:
    ------
    img: tiff img full shape (H,W,C)
    img_scale: scale to resize original image
    img_min_score: filter image with min score (prevent empty tile)
    tile_size: size of the tile
    tile_avg_step: average step for tile creation
    '''
    
    half = tile_size//2
    #TODO: provare altre interpolazioni #INTER LINEAR ecc..
    img_small = cv2.resize(img, dsize=None, fx=img_scale, fy=img_scale, interpolation = cv2.INTER_AREA)
    
    structure = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
    structure = (structure[:, :, 1] > 32).astype(np.uint8)
    
    structure = structure.astype(np.float32)
    
    h, w, _ = img_small.shape
    
    #creo il linspace per "tagliare" l'immagine in tile
    coord_x = np.linspace(half, w-half, int(np.ceil((w-tile_size)/tile_avg_step)), dtype=int)
    coord_y = np.linspace(half, h-half, int(np.ceil((h-tile_size)/tile_avg_step)), dtype=int)
    
    coord = []
    reject = []
    for cy in coord_y:
        for cx in coord_x:
            #filtro via per le immagini che non contengono nulla
            curr_val = structure[cy-half:cy+half, cx-half:cx+half].mean()
            
            if curr_val>tile_min_score:
                coord.append([cx, cy, curr_val])
            else:
                reject.append([cx, cy, curr_val])
    
    tile_img = []
    for cx, cy, cv in coord:
        im = img_small[cy-half:cy+half,cx-half:cx+half]
        tile_img.append(im)
        
    return {
        'coord': coord,
        'reject': reject,
        'img': img_small,
        'img_tile': tile_img,
        'structure': structure,
    }

def to_mask(tile, tile_coord, img_height, img_width, tile_size, aggregate='mean'):
    
    '''
    to_mask()>np.array: funzione che permette di creare una mask con dimensioni pari all'immagine
        ridimensionata tramite l'utilizzo di un Gaussian filter.

    tile: lista dei tile per quella maschera
    tile_coord: lista delle coordinate per i tile della maschera
    img_height: altezza dell'immagine dopo resize
    img_width: larghezza dell'immagine dopo resize
    tile_size: dimensione dei tile creati con to_tile
    '''
    half = tile_size//2
    mask  = np.zeros((img_height, img_width), np.float32)
    count = np.zeros((img_height, img_width), np.float32)

    w = np.ones((tile_size,tile_size), np.float32)
    
    #Creo un filtro gaussiano
    y,x = np.mgrid[-half:half,-half:half]
    y = half-abs(y)
    x = half-abs(x)
    w = np.minimum(x,y)
    w = w/w.max()#*2.5
    w = np.minimum(w,1)

    for t, (cx, cy, cv) in enumerate(tile_coord):
        mask [cy - half:cy + half, cx - half:cx + half] += tile[t]*w
        count[cy - half:cy + half, cx - half:cx + half] += w
           # see unet paper for "Overlap-tile strategy for seamless segmentation of arbitrary large images"
    m = (count != 0)
    mask[m] /= count[m]

    return mask