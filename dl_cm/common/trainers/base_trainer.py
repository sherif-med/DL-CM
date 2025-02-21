import pytorch_lightning as pl

from dl_cm.common import DLCM
from dl_cm.common.trainers.callbacks import CallbacksFactory, baseCallback
from dl_cm.common.trainers.loggers import BaseLogger, LoggersFactory
from dl_cm.common.trainers.plugins import BasePlugin, PluginsFactory
from dl_cm.common.typing import namedEntitySchema
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

TRAINERS_REGISTERY = Registry("Trainers")


class BaseTrainer(pl.Trainer, DLCM):
    @staticmethod
    def registry() -> Registry:
        return TRAINERS_REGISTERY

    def __init__(
        self,
        params: dict,
        callbacks: list[namedEntitySchema | baseCallback] = [],
        loggers: list[namedEntitySchema | BaseLogger] = [],
        plugins: list[namedEntitySchema | BasePlugin] = [],
        *args,
        **kwargs,
    ) -> None:
        callbacks = CallbacksFactory.create(callbacks)
        loggers = LoggersFactory.create(loggers)
        plugins = PluginsFactory.create(plugins)
        params: dict = params | {
            "callbacks": callbacks,
            "logger": loggers,
            "plugins": plugins,
        }
        DLCM.__init__(self, *args, **kwargs)
        pl.Trainer.__init__(self, **params)


class TrainerFactory(BaseFactory[BaseTrainer]):
    @staticmethod
    def base_class() -> type[BaseTrainer]:
        return BaseTrainer
