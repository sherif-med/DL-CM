from collections.abc import Callable

from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

FUNCTIONS_REGISTERY = Registry("Functions")


class FunctionsFactory(BaseFactory[Callable]):
    @staticmethod
    def base_class() -> Callable:
        return Callable
