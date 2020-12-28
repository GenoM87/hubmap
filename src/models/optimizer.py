import torch

from config import _C as cfg

def make_optimizer(model, cfg):

    assert cfg.SOLVER.OPTIMIZER in ['Adam', 'SGD'], 'Nome optimizer non riconosciuto!'

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