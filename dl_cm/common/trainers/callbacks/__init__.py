from dl_cm.utils.registery import Registry
from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.common.typing import namedEntitySchema
import pydantic as pd

CALLBACKS_REGISTERY = Registry("Callbacks")

class baseCallback(DLCM):

    @staticmethod
    def config_schema() -> pd.BaseModel:
        return namedEntitySchema
    
    @staticmethod
    def registry() -> Registry:
        return CALLBACKS_REGISTERY
    
    def __init__(self, config: dict) -> None:
        super().__init__(**config)

class CallbacksFactory(BaseFactory):
    
    @staticmethod
    def base_class()-> type:
        return baseCallback

from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.callbacks import LearningRateMonitor

CALLBACKS_REGISTERY.register(TQDMProgressBar)
CALLBACKS_REGISTERY.register(ModelCheckpoint)
CALLBACKS_REGISTERY.register(LearningRateFinder)
CALLBACKS_REGISTERY.register(LearningRateMonitor)
