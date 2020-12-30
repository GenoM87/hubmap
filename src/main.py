import sys, os, time, logging, datetime
from pathlib import Path

from engine.fitter import Fitter
from config import _C as cfg
from models.model import build_model

from data_builder import build_valid_loader, build_train_loader

#Creo lla directory per l'esperimento
path_exp = os.path.join(
    cfg.PROJECT_DIR, 'experiments', cfg.MODEL.NAME, str(datetime.date.today())
)

Path(path_exp).mkdir(parents=True, exist_ok=True)

#Istanzio il logger
path_logger = os.path.join(
    path_exp, f'train-{datetime.datetime.now()}.log'
)

logging.basicConfig(filename=path_logger, level=logging.DEBUG)
logger = logging.getLogger()

if __name__ == "__main__":
    
    model = build_model(cfg)
    train_loader = build_train_loader(cfg)
    valid_loader = build_valid_loader(cfg)

    engine = Fitter(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=valid_loader,
        logger=logger,
        exp_path=path_exp
    )

    #engine.train()
    engine.final_check()