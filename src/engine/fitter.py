import os, sys, time, warnings

import pandas as import pd
import numpy as np
import torch

from models.optimizer import make_optimizer
from models.scheduler import make_scheduler

class Fitter:

    def __init__(self, model, cfg, train_loader, val_loader, logger):

        self.model = model.to(cfg.DEVICE)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        self.optimizer = get_
