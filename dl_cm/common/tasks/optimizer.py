from dl_cm.utils.registery import Registry
from torch.optim import Optimizer

OPTIMIZER_REGiSTERY = Registry("Optimizer")

def load_optimizer_from_config(model_params, optimizer_config:dict)-> Optimizer:
    """"""
    optimizer_cls = OPTIMIZER_REGiSTERY.get(optimizer_config.get("name"))
    optimizer_params = optimizer_config.get("params")
    return optimizer_cls(model_params, **optimizer_params)
    

from torch.optim import SGD, Adam, AdamW

OPTIMIZER_REGiSTERY.register(SGD, "SGD")
OPTIMIZER_REGiSTERY.register(Adam, "Adam")
OPTIMIZER_REGiSTERY.register(AdamW, "AdamW")
