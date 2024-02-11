from src.utils.registery import Registry
from torch.optim import Optimizer

OPTIMIZER_REGiSTERY = Registry("Optimizer")

def load_optimizer_from_config(model_params, optimizer_config:dict)-> Optimizer:
    """"""
    optimizer_cls = OPTIMIZER_REGiSTERY.get(optimizer_config.get("name"))
    optimizer_params = optimizer_config.get("params")
    return optimizer_cls(model_params, **optimizer_params)
    