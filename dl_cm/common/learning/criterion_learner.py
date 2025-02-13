from dl_cm.common.learning.base_learner import BaseLearner
from dl_cm.utils.ppattern.data_validation import validationMixin
from dl_cm.common.tasks.criterion import CritireonFactory
import pydantic as pd
from dl_cm.common.tasks.criterion import BaseLoss
from dl_cm.common.typing import lossOutputStruct
from dl_cm.common.learning.optimizable_learner import OptimizableLearner
from abc import ABC, abstractmethod

class CriterionLearner(OptimizableLearner, validationMixin, ABC):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            critireon: dict
        return ValidConfig

    def __init__(self, learner_config:dict):
        validationMixin.__init__(self, learner_config)
        super().__init__(self, learner_config)
        self.criterion : BaseLoss = CritireonFactory.create(self.config.get("critireon"))
    
    @abstractmethod
    def criteron_step(self, *args, **kwargs) -> lossOutputStruct:
        pass
    