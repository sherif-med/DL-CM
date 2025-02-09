from dl_cm.utils.registery import Registry
from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory

OPTIMIZER_REGiSTERY = Registry("Optimizer")
LR_SCHEDULER_REGiSTERY = Registry("Lr scheduler")

class BaseOptimizer(DLCM):

    @staticmethod
    def registry() -> Registry:
        return OPTIMIZER_REGiSTERY
    
class OptimizerFactory(BaseFactory):
    
    @staticmethod
    def base_class()-> type:
        return BaseOptimizer

class BaseLrScheduler(DLCM):

    @staticmethod
    def registry() -> Registry:
        return LR_SCHEDULER_REGiSTERY
    
class LrSchedulerFactory(BaseFactory):
    
    @staticmethod
    def base_class()-> type:
        return BaseLrScheduler

import torch.optim as optim
for name in dir(optim):
    attr = getattr(optim, name)
    if isinstance(attr, type) and issubclass(attr, optim.Optimizer) and attr.__module__ == optim.__name__:
        OPTIMIZER_REGiSTERY.register(attr)

import torch.optim.lr_scheduler as lr_scheduler
for name in dir(lr_scheduler):
    attr = getattr(lr_scheduler, name)
    if isinstance(attr, type) and issubclass(attr, lr_scheduler._LRScheduler):
        LR_SCHEDULER_REGiSTERY.register(attr)
