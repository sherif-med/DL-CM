from dl_cm.common.typing import StepInputStruct, StepOutputStruct
from dl_cm.utils.ppattern.data_validation import validationMixin
from dl_cm.common.models import ModelsFactory, BaseModel
from dl_cm.common.typing import namedEntitySchema
import pydantic as pd
from dl_cm.utils.registery import Registry
from dl_cm.common import DLCM
from dl_cm.common.learning import LEARNERS_REGISTERY

class BaseLearner(validationMixin, DLCM):

    @staticmethod
    def registry() -> Registry:
        return LEARNERS_REGISTERY

    def config_schema(cls)-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            model: namedEntitySchema = None
        return ValidConfig

    def __init__(self, config) -> None:
        validationMixin.__init__(self, config)
        self.config = config
        self._model : BaseModel = ModelsFactory.create(config.get("model"))
    
    @property
    def model(self):
        return self._model

    def step(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        raise NotImplementedError