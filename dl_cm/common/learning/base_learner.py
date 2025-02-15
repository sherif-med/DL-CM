from dl_cm.common.typing import StepInputStruct, StepOutputStruct
from dl_cm.utils.ppattern.data_validation import validationMixin
from dl_cm.common.models import ModelsFactory, BaseModel
from dl_cm.common.typing import namedEntitySchema
import pydantic as pd
from dl_cm.utils.registery import Registry
from dl_cm.common import DLCM
from dl_cm.common.learning import LEARNERS_REGISTERY
import torch

class BaseLearner(torch.nn.Module, validationMixin, DLCM):

    @staticmethod
    def registry() -> Registry:
        return LEARNERS_REGISTERY

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(namedEntitySchema):
            model: namedEntitySchema = None
        return ValidConfig

    def __init__(self, config: dict) -> None:
        validationMixin.__init__(self, config)
        super().__init__()
        self._model : BaseModel = ModelsFactory.create(config.get("model"))

    @property
    def model(self):
        return self._model

    def forward(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        return self.model(batch["inputs"], *args, **kwargs)