#https://github.com/mpariente/Ranger-Deep-Learning-Optimizer/tree/master/pytorch_ranger

import torch
from pytorch_ranger import Ranger, RangerQH, RangerVA

from config import _C as cfg

def make_optimizer(model, cfg):

    assert cfg.SOLVER.OPTIMIZER in ['Adam', 'SGD', 'Ranger', 'RangerQH', 'RangerALR'], 'Nome optimizer non riconosciuto!'

    if cfg.SOLVER.OPTIMIZER == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.SOLVER.LR,
            weight_decay= cfg.SOLVER.WEIGHT_DECAY,
            betas=cfg.SOLVER.BETAS,
            amsgrad=cfg.SOLVER.AMSGRAD
        )
    elif cfg.SOLVER.OPTIMIZER=='SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.SOLVER.LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            nesterov=cfg.SOLVER.NESTEROS
        )
    
    elif cfg.SOLVER.OPTIMIZER=='Ranger':
        return Ranger(
            model.parameters(),
            lr=cfg.SOLVER.LR
            )
    elif cfg.SOLVER.OPTIMIZER=='RangerQH':
        return RangerQH(
            model.parameters(),
            lr=cfg.SOLVER.LR
            )
    elif cfg.SOLVER.OPTIMIZER=='RangerALR':
        return RangerVA(
            model.parameters(),
            lr=cfg.SOLVER.LR
            )