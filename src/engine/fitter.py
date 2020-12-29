import os, sys, time, warnings, datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

from utils import create_tile_v2
from data_builder.transforms import get_valid_transform
from .average import AverageMeter
from models.optimizer import make_optimizer
from models.scheduler import make_scheduler
from models.loss import binary_xloss, dice_coefficient

class Fitter:
    def __init__(self, model, cfg, train_loader, val_loader, logger, exp_path):
        
        print(datetime.date.today())
        self.experiment_path =  exp_path
        os.makedirs(self.experiment_path, exist_ok=True)

        self.model = model.to(cfg.DEVICE)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

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

        self.logger.info(f'Avvio training {time.time()} con i seguenti parametri:')
        self.logger.info(self.cfg)

    def train(self):
        #Start training loop
        for epoch in range(0, self.cfg.SOLVER.NUM_EPOCHS):

            t = time.time()
            summary_loss = self.train_one_epoch()
            self.logger.info(
                f'''[RESULT]: Train. Epoch: {self.epoch},
                summary_loss: {summary_loss.avg:.5f}, 
                time: {(time.time() - t):.3f}'''
            )

            valid_loss, valid_dice, best_thr = self.validate()

            self.scheduler.step(valid_dice)

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
            targets = masks.to(self.cfg.DEVICE)

            y_hat = self.model(imgs).squeeze(1)
            loss = binary_xloss(y_hat, targets)

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
                y_hat = self.model(imgs).squeeze(1)
                loss = binary_xloss(y_hat, targets)

                prob = torch.sigmoid(y_hat)

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
        self.model.eval()
        t = time.time()
        
        tmsf = get_valid_transform(self.cfg)

        df = pd.read_csv(
            os.path.join(self.cfg.DATA_DIR, 'train.csv')
        )
        for img_id in self.cfg.DATASET.VALID_ID:

            tile = create_tile_v2(
                img_id, df, self.cfg
            )

            tile_image = tile['img_tile']
            tile_image = np.stack(tile_image)[..., ::-1]
            print(tile_image.shape)
            tile_image = np.ascontiguousarray(tile_image.transpose(0,3,1,2))
            print(tile_image.shape)

            tile_probability = []
            batch = np.array_split(tile_image, len(tile_image)//4)
            batch = tmsf(batch)
            for t, m in enumerate(batch):
                m
                preds = self.model(batch).squeeze(1)



