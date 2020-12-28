import os, sys, time, warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

from .average import AverageMeter
from models.optimizer import make_optimizer
from models.scheduler import make_scheduler
from models.loss import binary_xloss


class Fitter:

    def __init__(self, model, cfg, train_loader, val_loader, logger, exp_path):
        
        self.experiment_path =  exp_path

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
            self.logger.info(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')

            valid_loss, valid_dice = self.validate()

            self.scheduler.step(valid_dice)

            self.logger.info( f'[RESULT]: Val. Epoch: {self.epoch}, Best Score Threshold: {self.best_threshold:.2f}, Best Score: {valid_dice:.5f}, time: {(time.time() - t):.5f}')
            if valid_dice > self.val_score:
                self.model.eval()
                self.save(self.experiment_path)
                self.val_score = valid_dice

    def train_one_epoch(self):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()

        train_loader = tqdm(self.train_loader, total=len(self.train_loader), desc='Training')

        for step, (imgs, targets) in enumerate(train_loader):
            
            self.optimizer.zero_grad()
            batch_size = imgs.shape[0]

            imgs = imgs.to(self.cfg.DEVICE)
            targets = targets.to(self.cfg.DEVICE)

            y_hat = self.model(imgs).squeeze(1)
            loss = binary_xloss(y_hat, targets)

            loss.backward()
            self.optimizer.step()

            summary_loss.update(loss.detach().cpu().item(), batch_size)

            train_loader.set_description(
                f'Train Step {step}/{len(self.train_loader)}, ' + \
                f'Learning rate {self.optimizer.param_groups[0]["lr"]}, ' + \
                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}'
            )


        return summary_loss

    def validate(self):
        self.model.eval()
        t = time.time()
        summary_loss = AverageMeter()


        return val_loss

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


