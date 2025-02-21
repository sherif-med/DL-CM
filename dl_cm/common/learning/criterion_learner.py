from abc import ABC, abstractmethod

from dl_cm.common.learning.optimizable_learner import OptimizableLearner
from dl_cm.common.tasks.criterion import BaseLoss, CritireonFactory
from dl_cm.common.typing import lossOutputStruct, namedEntitySchema


class CriterionLearner(OptimizableLearner, ABC):
    def __init__(self, criterion: BaseLoss | namedEntitySchema, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion: BaseLoss = CritireonFactory.create(criterion)

    @abstractmethod
    def criteron_step(self, *args, **kwargs) -> lossOutputStruct:
        pass
