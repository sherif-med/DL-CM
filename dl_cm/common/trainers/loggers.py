from dl_cm.utils.registery import Registry
from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.ppattern.data_validation import validationMixin

LOGGERS_REGISTERY = Registry("Loggers")

class BaseLogger(DLCM, validationMixin):

    @staticmethod
    def registry() -> Registry:
        return LOGGERS_REGISTERY
    
class LoggersFactory(BaseFactory):
    
    @staticmethod
    def base_class()-> type:
        return BaseLogger

import pytorch_lightning.loggers as pl_loggers
for name in dir(pl_loggers):
    attr = getattr(pl_loggers, name)
    if isinstance(attr, type) and issubclass(attr, pl_loggers.Logger) and attr.__module__ == pl_loggers.__name__:
        LOGGERS_REGISTERY.register(attr)