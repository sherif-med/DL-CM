from . import TRANSFORMATION_REGISTRY
from dl_cm.common import DLCM
from dl_cm.common.typing import Registry
from dl_cm.utils.ppattern.factory import BaseFactory

class GeneralTransformation(DLCM):
    
    @staticmethod
    def registry() -> Registry:
        return TRANSFORMATION_REGISTRY

class GeneralTransformationFactory(BaseFactory[GeneralTransformation]):
    
    @staticmethod
    def base_class()-> type[GeneralTransformation]:
        return GeneralTransformation
    

class GeneralIrrevirsibleTransformation(GeneralTransformation):
    
    def __call__(self, item, **kwargs):        
        return self.__fwd__(item, **kwargs)
    
    def __fwd__(self, *args):
        raise NotImplementedError

class GeneralRevrsibleTransformation(GeneralTransformation):
    
    def __init__(self, fwdfn, rwdfn):
        self.__fwd__fn__ = fwdfn
        self.__rwd__fn__ = rwdfn
        
    def __call__(self, item, **kwargs):        
        if kwargs.get("reverse", False):
            kwargs.pop("reverse")
            return self.__rwd__(item, **kwargs)
        else:
            return self.__fwd__(item, **kwargs)
    
    def __fwd__(self, *args):
        return self.__fwd__fn__(*args)
    
    def __rwd__(self, *args):
        return self.__rwd__fn__(*args)