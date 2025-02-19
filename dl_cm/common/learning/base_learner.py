import torch

from dl_cm.common import DLCM
from dl_cm.common.learning import LEARNERS_REGISTERY
from dl_cm.common.models import BaseModel, ModelsFactory
from dl_cm.common.typing import StepInputStruct, StepOutputStruct, namedEntitySchema
from dl_cm.utils.registery import Registry


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
