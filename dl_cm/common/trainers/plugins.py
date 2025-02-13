from dl_cm.utils.registery import Registry
from dl_cm.utils.ppattern.data_validation import validationMixin
import pytorch_lightning as pl
import pydantic as pd
from dl_cm.common.typing import namedEntitySchema
from dl_cm.utils.ppattern.factory import BaseFactory

PLUGINS_REGISTERY = Registry("Plugins")

class BasePlugin(pl.TrainerPlugin):
    @staticmethod
    def registry() -> Registry:
        return PLUGINS_REGISTERY

    @staticmethod
    def config_schema() -> pd.BaseModel:
        return namedEntitySchema

    def __init__(self, config: dict) -> None:
        validationMixin.__init__(self, config)
        super().__init__(**config)

class PluginsFactory(BaseFactory[BasePlugin]):
    @staticmethod
    def base_class()-> type[BasePlugin]:
        return BasePlugin