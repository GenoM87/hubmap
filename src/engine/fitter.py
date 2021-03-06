import os, sys, time, warnings, datetime, gc
from collections import OrderedDict

import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

import torch
import torch.nn.functional as F
import albumentations as A
import rasterio

from utils import create_tile_v2, rle2mask, to_mask, global_shift_mask
from data_builder.transforms import get_valid_transform
from .average import AverageMeter
from models.optimizer import make_optimizer
from models.scheduler import make_scheduler
from models.loss import binary_xloss, dice_coefficient, dice_coeff
from models.SegLoss.losses_pytorch.dice_loss import SoftDiceLoss

class Fitter:
    def __init__(self, model, cfg, train_loader, val_loader, logger, exp_path):
        
        self.experiment_path =  exp_path
        os.makedirs(self.experiment_path, exist_ok=True)

        self.model = model.to(cfg.DEVICE)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.criterion = SoftDiceLoss()

        self.optimizer = make_optimizer(
            self.model, self.cfg
        )

        self.scheduler = make_scheduler(
            self.optimizer, 
            self.cfg,
            self.train_loader
        )

        self.epoch = 0
        self.val_score = 0
        self.best_threshold = 0

        self.logger.info(f'Avvio training {datetime.datetime.now()} con i seguenti parametri:')
        self.logger.info(self.cfg)

    def train(self):
        #Start training loop
        for epoch in range(self.epoch, self.cfg.SOLVER.NUM_EPOCHS):

            if epoch < self.cfg.SOLVER.WARMUP_EPOCHS:
                #Create increasing lr
                lr = np.linspace(
                    start=self.cfg.SOLVER.MIN_LR, 
                    stop=self.cfg.SOLVER.LR, 
                    num=self.cfg.SOLVER.WARMUP_EPOCHS
                )

                for g in self.optimizer.param_groups:
                    g['lr'] = lr[epoch]
                
                self.logger.info(f'[TRAIN]WARMUP: Increasing learning rate to {lr[epoch]}')

            t = time.time()
            summary_loss = self.train_one_epoch()
            self.logger.info(
                f'''[RESULT]: Train. Epoch: {self.epoch},
                summary_loss: {summary_loss.avg:.5f}, 
                time: {(time.time() - t):.3f}'''
            )

            valid_loss, valid_dice, best_thr = self.validate()

            if self.cfg.SOLVER.SCHEDULER == 'ReduceLROnPlateau':
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()

            self.logger.info(
                f'''[RESULT]: Val. Epoch: {self.epoch},
                validation_loss: {valid_loss.avg:.5f},
                Best Score Threshold: {self.best_threshold:.2f}, 
                Best Score: {valid_dice:.5f}, 
                time: {(time.time() - t):.3f}'''
            )
            self.epoch += 1
            if valid_dice > self.val_score:
                self.model.eval()
                self.save(
                    os.path.join(self.experiment_path, f'unet_best.ckpt'))
                self.val_score = valid_dice
                self.best_threshold = best_thr

    def train_one_epoch(self):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()

        train_loader = tqdm(self.train_loader, total=len(self.train_loader), desc='Training')

        for step, (imgs, masks) in enumerate(train_loader):
            
            self.optimizer.zero_grad()
            batch_size = imgs.shape[0]

            imgs = imgs.to(self.cfg.DEVICE)
            targets = masks.to(self.cfg.DEVICE).unsqueeze(1)

            prob = self.model(imgs)
            
            loss = self.criterion(prob, targets)

            loss.backward()
            self.optimizer.step()

            summary_loss.update(loss.detach().cpu().item(), batch_size)

            train_loader.set_description(
                f'Train Step {step}/{len(self.train_loader)}, ' + \
                f'Learning rate {self.optimizer.param_groups[0]["lr"]}, ' + \
                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                f'time: {(time.time() - t):.3f}'
            )

        return summary_loss

    def validate(self):
        self.model.eval()
        t = time.time()
        summary_loss = AverageMeter()

        val_loader = tqdm(self.val_loader, total=len(self.val_loader), desc='Valid')

        valid_probability = []
        valid_mask = []
        for step, (imgs, masks) in enumerate(val_loader):

            targets = masks.to(self.cfg.DEVICE)
            imgs = imgs.to(self.cfg.DEVICE)
            batch_size = imgs.shape[0]

            with torch.no_grad():
                prob = self.model(imgs).squeeze()

                loss = self.criterion(prob, targets)

            summary_loss.update(loss, batch_size)
            valid_probability.append(prob.detach().cpu().numpy())
            valid_mask.append(targets.detach().cpu().numpy())

            val_loader.set_description(
                f'Valid Step {step}/{len(self.val_loader)}, ' + \
                f'Learning rate {self.optimizer.param_groups[0]["lr"]}, ' + \
                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                f'time: {(time.time() - t):.3f}'
            )

        probability = np.concatenate(valid_probability)
        mask = np.concatenate(valid_mask)

        act_dice = 0
        best_dice = 0
        for thr in np.linspace(0, 1, num=20):
            pred = (probability>thr).astype(np.uint8)
            act_dice = dice_coefficient(pred, mask)
            if act_dice>best_dice:
                self.logger.info(
                    f'[VALID]Epoch: {self.epoch} Found best dice at thr {thr}: {act_dice}'
                )
                best_thr = thr
                best_dice = act_dice

        self.best_threshold = best_thr
        return summary_loss, best_dice, best_thr

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_threshold': self.best_threshold,
            'val_score': self.val_score,
            'epoch': self.epoch,
        }, path)

    def save_model(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_threshold': self.best_threshold,
            'val_score': self.val_score,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_threshold = checkpoint['best_threshold']
        self.val_score = checkpoint['val_score']
        self.epoch = checkpoint['epoch'] + 1

    def final_check(self):
        
        ckpt = torch.load(self.cfg.MODEL.CHECKPOINT_PATH)
        if 'model_state_dict' in list(ckpt.keys()):
            self.load(self.cfg.MODEL.CHECKPOINT_PATH)
        else:
            state_dict = ckpt['state_dict']
            state = OrderedDict([(key.split("model.")[-1], state_dict[key]) for key in state_dict])
            self.model.load_state_dict(state)
            del ckpt, state, state_dict
            
        gc.collect()

        self.model = self.model.to(self.cfg.DEVICE)

        self.model.eval()
        t = time.time()
        
        norm = A.Normalize()

        df = pd.read_csv(
            os.path.join(self.cfg.DATA_DIR, 'train.csv')
        )

        for img_id in self.cfg.DATASET.VALID_ID:
            
            #Tile creation per l'immagine phase=valid per usare i parametri di test
            tile = create_tile_v2(
                img_id, df, self.cfg, phase='valid'
            )

            path_img = os.path.join(
                self.cfg.DATA_DIR, 'train', img_id+'.tiff'
            )

            identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

            dataset = rasterio.open(path_img, transform=identity, num_threads = 'all_cpus',)
            h, w = dataset.shape

            encoding = df[df['id']==img_id]['encoding'].values[0]
            
            #CREO LA MASCHERA E RIDEFINISCO h,w
            mask = rle2mask(encoding, (w, h))
            mask = cv2.resize(
                mask, dsize=None, 
                fx=self.cfg.DATASET.IMG_SCALE, 
                fy=self.cfg.DATASET.IMG_SCALE, 
                interpolation=cv2.INTER_AREA
            )
            h, w = mask.shape

            tile_image = tile['img_tile']
            tile_image = np.stack(tile_image)[..., ::-1]
            tile_image = norm(image=tile_image)['image']
            tile_image = np.ascontiguousarray(tile_image.transpose(0,3,1,2))

            batch = np.array_split(tile_image, len(tile_image)//4)
    
            tile_prob = []
            #itero per tutti i batch
            for num, imgs in enumerate(batch):
                imgs = torch.from_numpy(imgs).to(self.cfg.DEVICE)
                p = []
                with torch.no_grad():
                    #plain image
                    y_hat = self.model(imgs)
                    p.append(torch.sigmoid(y_hat))
                    
                    #horizontal flip
                    y_hat = self.model(imgs.flip(dims=(2,)))
                    p.append(torch.sigmoid(y_hat.flip(dims=(2,))))
                    
                    #vertical flip
                    y_hat = self.model(imgs.flip(dims=(3,)))
                    p.append(torch.sigmoid(y_hat.flip(dims=(3,))))

                p = torch.stack(p).mean(0)
                tile_prob.append(p.data.detach().cpu().numpy())
            
            tile_prob = np.concatenate(tile_prob).squeeze()
            mask_pred = to_mask(
                tile_prob,
                tile['coord'],
                h,
                w,
                self.cfg.DATASET.TEST_TILE_SIZE
            )

            predict = (mask_pred>self.best_threshold).astype(np.float32)
            base_dice = dice_coefficient(predict, mask)

            self.logger.warning(f'''
                [VALID]Immagine: {img_id}, Dice Coeff con threshold {self.best_threshold}: {base_dice}
            ''')
            self.logger.warning(f'[VALID]Avvio ricerca best thr per immagine {img_id}')

            for thr in np.linspace(0, 1, 21):
                predict = (mask_pred>thr).astype(np.float32)
                dice = dice_coefficient(predict, mask)
                if dice>base_dice:
                    base_dice=dice
                    self.best_threshold = thr
            
            self.logger.warning(f'''
                [VALID]Immagine: {img_id}, Dice Coeff finale con threshold {self.best_threshold}: {base_dice}
            ''')

            #Scrittura con previsioni finali
            predict = (mask_pred>self.best_threshold).astype(np.float32)

            #Scrittura delle immagini finali
            cv2.imwrite(
                os.path.join(self.experiment_path, img_id+'.probability.png'), mask_pred*255
            )

            cv2.imwrite(
                os.path.join(self.experiment_path, img_id+'.predict.png'), predict*255
            )

            cv2.imwrite(
                os.path.join(self.experiment_path, img_id+'.mask.png'), mask*255
            )

    def compute_shift(self):

        ckpt = torch.load(self.cfg.MODEL.CHECKPOINT_PATH)
        if 'model_state_dict' in list(ckpt.keys()):
            self.load(self.cfg.MODEL.CHECKPOINT_PATH)
        else:
            state_dict = ckpt['state_dict']
            state = OrderedDict([(key.split("model.")[-1], state_dict[key]) for key in state_dict])
            self.model.load_state_dict(state)
            del ckpt, state, state_dict
            
        gc.collect()

        self.model = self.model.to(self.cfg.DEVICE)

        self.model.eval()
        t = time.time()
        
        norm = A.Normalize()

        df = pd.read_csv(
            os.path.join(self.cfg.DATA_DIR, 'train.csv')
        )

        f = open(
            os.path.join(self.experiment_path, 'compute_shift.txt'), 'a'
        )

        for img_id in list(df['id']):
            
            #Tile creation per l'immagine phase=valid per usare i parametri di test
            tile = create_tile_v2(
                img_id, df, self.cfg, phase='valid'
            )

            path_img = os.path.join(
                self.cfg.DATA_DIR, 'train', img_id+'.tiff'
            )

            identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

            dataset = rasterio.open(path_img, transform=identity, num_threads = 'all_cpus',)
            h, w = dataset.shape

            encoding = df[df['id']==img_id]['encoding'].values[0]
            
            #CREO LA MASCHERA E RIDEFINISCO h,w
            mask = rle2mask(encoding, (w, h))
            mask = cv2.resize(
                mask, dsize=None, 
                fx=self.cfg.DATASET.IMG_SCALE, 
                fy=self.cfg.DATASET.IMG_SCALE, 
                interpolation=cv2.INTER_AREA
            )
            h, w = mask.shape

            tile_image = tile['img_tile']
            tile_image = np.stack(tile_image)[..., ::-1]
            tile_image = norm(image=tile_image)['image']
            tile_image = np.ascontiguousarray(tile_image.transpose(0,3,1,2))

            batch = np.array_split(tile_image, len(tile_image)//4)
    
            tile_prob = []
            #itero per tutti i batch
            for num, imgs in enumerate(batch):
                imgs = torch.from_numpy(imgs).to(self.cfg.DEVICE)
                p = []
                with torch.no_grad():
                    #plain image
                    y_hat = self.model(imgs)
                    p.append(torch.sigmoid(y_hat))
                    
                    #horizontal flip
                    y_hat = self.model(imgs.flip(dims=(2,)))
                    p.append(torch.sigmoid(y_hat.flip(dims=(2,))))
                    
                    #vertical flip
                    y_hat = self.model(imgs.flip(dims=(3,)))
                    p.append(torch.sigmoid(y_hat.flip(dims=(3,))))

                p = torch.stack(p).mean(0)
                tile_prob.append(p.data.detach().cpu().numpy())
            
            tile_prob = np.concatenate(tile_prob).squeeze()
            mask_pred = to_mask(
                tile_prob,
                tile['coord'],
                h,
                w,
                self.cfg.DATASET.TEST_TILE_SIZE
            )

            predict = (mask_pred>self.best_threshold).astype(np.float32)

            shift_x = np.linspace(-25, 25, 51, dtype=int)
            shift_y = np.linspace(-25, 25, 51, dtype=int)

            best_score = dice_coefficient(predict, mask)

            f.write(f'Starting search for image: {img_id} with dice: {best_score}')
            for sh_x in shift_x:
                for sh_y in shift_y:
                    pred_shft = global_shift_mask(predict, y_shift=sh_y, x_shift=sh_x)
                    dice_shft = dice_coefficient(pred_shft, mask)

                    if dice_shft>=best_score:
                        f.write(f'Better shifting found {img_id} - x: {sh_x}, y: {sh_y}, dice: {dice_shft:.4f} \n')
                        best_score = dice_shft

        f.close()



