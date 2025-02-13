from dl_cm.utils.registery import Registry
from dl_cm.utils.ppattern.factory import BaseFactory
from collections.abc import Callable

FUNCTIONS_REGISTERY = Registry("Functions")

class FunctionsFactory(BaseFactory):
    
    @staticmethod
    def base_class()-> type:
        return Callable