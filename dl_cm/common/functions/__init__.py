from collections.abc import Callable

from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

FUNCTIONS_REGISTERY = Registry("Functions")

class BaseCallable(DLCM):
    @staticmethod
    def registry() -> Registry:
        return FUNCTIONS_REGISTERY

class FunctionsFactory(BaseFactory[BaseCallable]):
    @staticmethod
    def base_class(similar=False) -> type[BaseCallable]:
        if similar:
            return (BaseCallable,)
        return BaseCallable
