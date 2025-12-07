from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

PLUGINS_REGISTERY = Registry("Plugins")


class BasePlugin(DLCM):
    @staticmethod
    def registry() -> Registry:
        return PLUGINS_REGISTERY

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class PluginsFactory(BaseFactory[BasePlugin]):
    @staticmethod
    def base_class(similar=False) -> type[BasePlugin]:
        if similar:
            return (BasePlugin,)
        return BasePlugin


import pytorch_lightning.plugins as pl_plugins

for name in dir(pl_plugins):
    attr = getattr(pl_plugins, name)
    if isinstance(attr, type) and attr.__module__ == pl_plugins.__name__:
        _ = DLCM.base_class_adapter(attr, base_cls=BasePlugin)
