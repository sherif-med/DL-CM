import pytorch_lightning as pl

from dl_cm.common.trainer.callbacks import CallbacksFactory, baseCallback
from dl_cm.common.trainer.loggers import BaseLogger, LoggersFactory
from dl_cm.common.trainer.plugins import BasePlugin, PluginsFactory
from dl_cm.common.typing import namedEntitySchema


def load_trainer(
    params: dict,
    callbacks: list[namedEntitySchema | baseCallback] = [],
    loggers: list[namedEntitySchema | BaseLogger] = [],
    plugins: list[namedEntitySchema | BasePlugin] = [],
) -> None:
    callbacks = CallbacksFactory.create(callbacks)
    loggers = LoggersFactory.create(loggers)
    plugins = PluginsFactory.create(plugins)
    params: dict = params | {
        "callbacks": callbacks,
        "logger": loggers,
        "plugins": plugins,
    }
    return pl.Trainer(**params)
