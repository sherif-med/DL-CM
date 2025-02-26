from dl_cm.common.learning import BaseLearner
from dl_cm.common.tasks.optimizer import (
    BaseLrScheduler,
    BaseOptimizer,
    LrSchedulerFactory,
    OptimizerFactory,
)
from dl_cm.common.typing import OneOrMany, namedEntitySchema


class OptimizableLearner(BaseLearner):
    def __init__(
        self,
        optimizers: OneOrMany[namedEntitySchema | BaseOptimizer],
        lr_schedulers: OneOrMany[namedEntitySchema | BaseLrScheduler],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._optimizers: BaseOptimizer = []
        if isinstance(optimizers, list):
            self._optimizers = optimizers
        else:
            self._optimizers = [optimizers]
        self._lr_schedulers: BaseLrScheduler = []
        if isinstance(lr_schedulers, list):
            self._lr_schedulers = lr_schedulers
        else:
            self._lr_schedulers = [lr_schedulers]

    def init_optimizer(
        self, optimizer_conf: namedEntitySchema | BaseOptimizer
    ) -> BaseOptimizer:
        if isinstance(optimizer_conf, BaseOptimizer):
            return optimizer_conf
        optimizer_conf["params"]["params"] = self.model.parameters()
        return OptimizerFactory.create(optimizer_conf)

    def init_lr_scheduler(
        self, lr_scheduler_conf: namedEntitySchema | BaseLrScheduler
    ) -> BaseLrScheduler:
        if isinstance(lr_scheduler_conf, BaseLrScheduler):
            return lr_scheduler_conf
        lr_scheduler_conf["params"]["optimizer"] = self.optimizers
        return LrSchedulerFactory.create(lr_scheduler_conf)

    @property
    def optimizers(self) -> list[BaseOptimizer]:
        if all(map(lambda o: isinstance(o, BaseOptimizer), self._optimizers)):
            return self._optimizers
        self._optimizers = [self.init_optimizer(o) for o in self._optimizers]
        return self._optimizers

    @property
    def lr_schedulers(self) -> list[BaseLrScheduler]:
        if all(map(lambda o: isinstance(o, BaseLrScheduler), self._lr_schedulers)):
            return self._lr_schedulers
        self._lr_schedulers = [self.init_lr_scheduler(o) for o in self._lr_schedulers]
        return self._lr_schedulers
