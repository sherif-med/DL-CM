from dl_cm.utils.registery import Registry
from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.common.typing import namedEntitySchema
import pydantic as pd
from dl_cm.utils.ppattern.data_validation import validationMixin

CALLBACKS_REGISTERY = Registry("Callbacks")

class baseCallback(DLCM, validationMixin):

    @staticmethod
    def config_schema() -> pd.BaseModel:
        return namedEntitySchema
    
    @staticmethod
    def registry() -> Registry:
        return CALLBACKS_REGISTERY
    
    def __init__(self, config: dict) -> None:
        validationMixin.__init__(self, config)

class CallbacksFactory(BaseFactory):
    
    @staticmethod
    def base_class()-> type:
        return baseCallback

from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.callbacks import LearningRateMonitor

from functools import partial
base_callback_adapter = partial(DLCM.base_class_adapter, base_cls=baseCallback)
CALLBACKS_REGISTERY.register(TQDMProgressBar, base_class_adapter=base_callback_adapter)
CALLBACKS_REGISTERY.register(ModelCheckpoint, base_class_adapter=base_callback_adapter)
CALLBACKS_REGISTERY.register(LearningRateFinder, base_class_adapter=base_callback_adapter)
CALLBACKS_REGISTERY.register(LearningRateMonitor, base_class_adapter=base_callback_adapter)
