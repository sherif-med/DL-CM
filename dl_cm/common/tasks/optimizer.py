from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

OPTIMIZER_REGiSTERY = Registry("Optimizer")
LR_SCHEDULER_REGiSTERY = Registry("Lr scheduler")


class BaseOptimizer(DLCM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def registry() -> Registry:
        return OPTIMIZER_REGiSTERY


class OptimizerFactory(BaseFactory[BaseOptimizer]):
    @staticmethod
    def base_class() -> type[BaseOptimizer]:
        return BaseOptimizer


class BaseLrScheduler(DLCM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def registry() -> Registry:
        return LR_SCHEDULER_REGiSTERY


class LrSchedulerFactory(BaseFactory):
    @staticmethod
    def base_class() -> type:
        return BaseLrScheduler


from functools import partial

base_optimizer_adapter = partial(DLCM.base_class_adapter, base_cls=BaseOptimizer)
base_lr_scheduler_adapter = partial(DLCM.base_class_adapter, base_cls=BaseLrScheduler)

import torch.optim as optim

for name in dir(optim):
    attr = getattr(optim, name)
    if (
        isinstance(attr, type)
        and issubclass(attr, optim.Optimizer)
        and attr.__module__ == optim.__name__
    ):
        OPTIMIZER_REGiSTERY.register(
            attr, name=name, base_class_adapter=base_optimizer_adapter
        )

import torch.optim.lr_scheduler as lr_scheduler

for name in dir(lr_scheduler):
    attr = getattr(lr_scheduler, name)
    if isinstance(attr, type) and issubclass(attr, lr_scheduler._LRScheduler):
        LR_SCHEDULER_REGiSTERY.register(
            attr, name=name, base_class_adapter=base_lr_scheduler_adapter
        )
