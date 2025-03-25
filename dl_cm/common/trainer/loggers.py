from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

LOGGERS_REGISTERY = Registry("Loggers")


class BaseLogger(DLCM):
    @staticmethod
    def registry() -> Registry:
        return LOGGERS_REGISTERY

    def __init__(self, *args, **config) -> None:
        super().__init__(*args, **config)


class LoggersFactory(BaseFactory[BaseLogger]):
    @staticmethod
    def base_class() -> type[BaseLogger]:
        return BaseLogger


import pytorch_lightning.loggers as pl_loggers

for name in dir(pl_loggers):
    attr = getattr(pl_loggers, name)
    if isinstance(attr, type) and issubclass(attr, pl_loggers.Logger):
        _ = DLCM.base_class_adapter(attr, base_cls=BaseLogger)
