from dl_cm.utils.registery import Registry
from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
from dl_cm.common.typing import namedEntitySchema

LOGGERS_REGISTERY = Registry("Loggers")

class BaseLogger(DLCM, validationMixin):

    @staticmethod
    def config_schema() -> pd.BaseModel:
        return namedEntitySchema
    
    @staticmethod
    def registry() -> Registry:
        return LOGGERS_REGISTERY
    
    def __init__(self, config: dict) -> None:
        validationMixin.__init__(self, config)
    
class LoggersFactory(BaseFactory[BaseLogger]):
    
    @staticmethod
    def base_class()-> type[BaseLogger]:
        return BaseLogger

from functools import partial
base_logger_adapter = partial(DLCM.base_class_adapter, base_cls=BaseLogger)

import pytorch_lightning.loggers as pl_loggers
for name in dir(pl_loggers):
    attr = getattr(pl_loggers, name)
    if isinstance(attr, type) and issubclass(attr, pl_loggers.Logger) and attr.__module__ == pl_loggers.__name__:
        LOGGERS_REGISTERY.register(attr, base_class_adapter=base_logger_adapter)