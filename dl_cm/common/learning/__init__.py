import torch

from dl_cm.common import DLCM
from dl_cm.common.models import BaseModel, ModelsFactory
from dl_cm.common.typing import StepInputStruct, StepOutputStruct, namedEntitySchema
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

LEARNERS_REGISTERY = Registry("Learners")


class BaseLearner(torch.nn.Module, DLCM):
    @staticmethod
    def registry() -> Registry:
        return LEARNERS_REGISTERY

    def __init__(self, model: BaseModel | namedEntitySchema, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model: BaseModel = ModelsFactory.create(model)

    @property
    def model(self):
        return self._model

    def forward(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        return self.model(batch["inputs"], *args, **kwargs)


class LearnersFactory(BaseFactory[BaseLearner]):
    @staticmethod
    def base_class() -> type[BaseLearner]:
        return BaseLearner


from .criterion_learner import CriterionLearner
from .equivariance_enforced_learner import equivarianceEnforcedLearner
from .extra_learners import MultiLearner, SequentialLearner, learnerWrapper
from .optimizable_learner import OptimizableLearner
from .supervised_learner import SupervisedLearner
from .test_learners import (
    SegmentationEquivarianceEnforcedLearner,
    acrossDimensionVarianceLoss,
)
from .unsupervised_learner import UnsupervisedLearner

__all__ = [
    "BaseLearner",
    "CriterionLearner",
    "LearnersFactory",
    "OptimizableLearner",
    "SupervisedLearner",
    "UnsupervisedLearner",
    "learnerWrapper",
    "MultiLearner",
    "SequentialLearner",
    "equivarianceEnforcedLearner",
    "SegmentationEquivarianceEnforcedLearner",
    "acrossDimensionVarianceLoss",
]
