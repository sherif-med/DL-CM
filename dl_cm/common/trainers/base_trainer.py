from dl_cm.common.trainers.callbacks import CallbacksFactory
from dl_cm.common.trainers.loggers import LoggersFactory
from dl_cm.common.trainers.plugins import PluginsFactory
from dl_cm.utils.registery import Registry
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.ppattern.data_validation import validationMixin
import pytorch_lightning as pl
import pydantic as pd

TRAINERS_REGISTERY = Registry("Trainers")

class BaseTrainer(pl.Trainer):
    @staticmethod
    def registry() -> Registry:
        return TRAINERS_REGISTERY

    @staticmethod
    def config_schema() -> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            params: dict = {}
            callbacks: list[dict] = []
            loggers: list[dict] = []
            plugins: list[dict] = []
        return ValidConfig

    def __init__(self, config: dict) -> None:
        validationMixin.__init__(self, config)
        callbacks = CallbacksFactory.create(config.get("callbacks", []))
        loggers = LoggersFactory.create(config.get("loggers", []))
        plugins = PluginsFactory.create(config.get("plugins", []))
        params: dict = config.get("params", {}) | {
            "callbacks": callbacks,
            "logger": loggers,
            "plugins": plugins
        }
        super().__init__(**params)

class TrainerFactory(BaseFactory):
    @staticmethod
    def base_class()-> type:
        return BaseTrainer