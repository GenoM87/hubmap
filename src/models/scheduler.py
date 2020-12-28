import torch

def make_scheduler(optimizer, cfg, train_loader):

    if cfg.SOLVER.SCHEDULER == 'CosineAnnealingWarmRestarts':
        number_of_iteration_per_epoch = len(train_loader)
        learning_rate_step_size = cfg.SOLVER.SCHEDULER_COS_CPOCH * number_of_iteration_per_epoch
        
        scheduler = getattr(torch.optim.lr_scheduler, cfg.SOLVER.SCHEDULER_NAME)(optimizer, T_0=learning_rate_step_size, T_mult=cfg.SOLVER.T_MUL)
        return scheduler

    elif cfg.SOLVER.SCHEDULER == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=cfg.SOLVER.SCHEDULER_MODE,
            factor=cfg.SOLVER.SCHEDULER_REDFACT,
            patience=cfg.SOLVER.SCHEDULER_PATIENCE
        )
    
    else:
        print('NOME SCHEDULER NON RICONOSCIUTO!')
