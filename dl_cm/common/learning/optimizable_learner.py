from typing import Union
import copy

from dl_cm.common import DLCM
from dl_cm.common.learning import BaseLearner
from dl_cm.common.typing import OneOrMany, namedEntitySchema
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry


class OptimizableLearner(BaseLearner):
    def __init__(
        self,
        optimizers: OneOrMany[Union[namedEntitySchema, "BaseOptimizer"]],
        lr_schedulers: OneOrMany[Union[namedEntitySchema, "BaseLrScheduler"]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._optimizers: BaseOptimizer = []
        optimizers = copy.deepcopy(optimizers) # Added to seperate optimizer params from config
        if isinstance(optimizers, list):
            self._optimizers = optimizers
        else:
            self._optimizers = [optimizers]
        self._lr_schedulers: BaseLrScheduler = []
        if isinstance(lr_schedulers, list):
            self._lr_schedulers = lr_schedulers
        elif isinstance(lr_schedulers, BaseLrScheduler):
            self._lr_schedulers = [lr_schedulers]

    def init_optimizer(
        self, optimizer_conf: Union[namedEntitySchema, "BaseOptimizer"]
    ) -> "BaseOptimizer":
        if isinstance(optimizer_conf, BaseOptimizer):
            return optimizer_conf
        optimizer_conf["params"]["params"] = self.model.parameters()
        return OptimizerFactory.create(optimizer_conf)

    def init_lr_scheduler(
        self, lr_scheduler_conf: Union[namedEntitySchema, "BaseLrScheduler"]
    ) -> "BaseLrScheduler":
        if isinstance(lr_scheduler_conf, BaseLrScheduler):
            return lr_scheduler_conf
        lr_scheduler_conf["params"]["optimizer"] = self.optimizers
        return LrSchedulerFactory.create(lr_scheduler_conf)

    @property
    def optimizers(self) -> list["BaseOptimizer"]:
        if all(map(lambda o: isinstance(o, BaseOptimizer), self._optimizers)):
            return self._optimizers
        self._optimizers = [self.init_optimizer(o) for o in self._optimizers]
        return self._optimizers

    @property
    def lr_schedulers(self) -> list["BaseLrScheduler"]:
        if all(map(lambda o: isinstance(o, BaseLrScheduler), self._lr_schedulers)):
            return self._lr_schedulers
        self._lr_schedulers = [self.init_lr_scheduler(o) for o in self._lr_schedulers]
        return self._lr_schedulers


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
    def base_class(similar=False) -> type[BaseOptimizer]:
        return BaseOptimizer


class BaseLrScheduler(DLCM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def registry() -> Registry:
        return LR_SCHEDULER_REGiSTERY


class LrSchedulerFactory(BaseFactory):
    @staticmethod
    def base_class(similar=False) -> type:
        return BaseLrScheduler


import torch.optim as optim

for name in dir(optim):
    attr = getattr(optim, name)
    if isinstance(attr, type) and issubclass(attr, optim.Optimizer):
        _ = DLCM.base_class_adapter(attr, base_cls=BaseOptimizer)

import torch.optim.lr_scheduler as lr_scheduler

for name in dir(lr_scheduler):
    attr = getattr(lr_scheduler, name)
    if isinstance(attr, type) and issubclass(attr, lr_scheduler._LRScheduler):
        _ = DLCM.base_class_adapter(attr, base_cls=BaseLrScheduler)
