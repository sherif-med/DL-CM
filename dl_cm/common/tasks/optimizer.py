from dl_cm.utils.registery import Registry
from torch.optim import Optimizer

OPTIMIZER_REGiSTERY = Registry("Optimizer")
LR_SCHEDULER_REGiSTERY = Registry("Lr scheduler")

def load_optimizer_from_config(model_params, optimizer_config:dict)-> Optimizer:
    """"""
    optimizer_cls = OPTIMIZER_REGiSTERY.get(optimizer_config.get("name"))
    optimizer_params = optimizer_config.get("params")
    return optimizer_cls(model_params, **optimizer_params)

def load_lr_scheduler_from_config(optimizer, lr_scheduler_config:dict):
    scheduler_cls = LR_SCHEDULER_REGiSTERY.get(lr_scheduler_config.get("name"))
    scheduler_params = lr_scheduler_config.get("params")
    return scheduler_cls(optimizer, **scheduler_params)

from torch.optim import SGD, Adam, AdamW

OPTIMIZER_REGiSTERY.register(SGD, "SGD")
OPTIMIZER_REGiSTERY.register(Adam, "Adam")
OPTIMIZER_REGiSTERY.register(AdamW, "AdamW")

from torch.optim.lr_scheduler import StepLR

LR_SCHEDULER_REGiSTERY.register(StepLR)
