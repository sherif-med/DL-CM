from dl_cm.common.tasks.optimizer import OptimizerFactory, BaseOptimizer, LrSchedulerFactory, BaseLrScheduler
from dl_cm.common.learning.base_learner import BaseLearner
import pydantic as pd
from dl_cm.common.typing import namedEntitySchema
from dl_cm.utils.ppattern.data_validation import validationMixin
import copy

class OptimizableLearner(BaseLearner, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(namedEntitySchema):
            optimizer: namedEntitySchema | list[namedEntitySchema] = None
            lr_scheduler: namedEntitySchema | list[namedEntitySchema] = None
        return ValidConfig

    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        super().__init__(config=config)
        self._optimizer : BaseOptimizer = None
        self._lr_scheduler : BaseLrScheduler = None
    
    @property
    def optimizer(self)-> list[BaseOptimizer]:
        if self._optimizer is None:
            optimizer_config = copy.copy(self.config["optimizer"])
            optimizer_config["params"]["params"] = self.model.parameters()
            self._optimizer = OptimizerFactory.create(optimizer_config)
        # ensure optimizer is a list
        if not isinstance(self._optimizer, list):
            self._optimizer = [self._optimizer]
        return self._optimizer
    
    @property
    def lr_scheduler(self)-> list[BaseLrScheduler]:
        if self._lr_scheduler is None:
            lr_scheduler_config = copy.copy(self.config["lr_scheduler"])
            lr_scheduler_config["params"]["optimizer"] = self.model.parameters()
            self._lr_scheduler = OptimizerFactory.create(lr_scheduler_config)
        # ensure lr_scheduler is a list
        if not isinstance(self._lr_scheduler, list):
            self._lr_scheduler = [self._lr_scheduler]
        return self._lr_scheduler
    
