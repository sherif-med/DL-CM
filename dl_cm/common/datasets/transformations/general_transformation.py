from . import TRANSFORMATION_REGISTRY
from dl_cm.config_loaders import load_named_entity
from dl_cm.utils.exceptions import OutOfTypesException

class GeneralTransformation:
    pass

class GeneralTransformationFactory:
    def __init__(self):
        pass

    @classmethod
    def create(cls, c_transform):
        if isinstance(c_transform, str):
            transformation_class = TRANSFORMATION_REGISTRY.get(c_transform)
            return transformation_class()
        elif isinstance(c_transform, dict):
            return load_named_entity(TRANSFORMATION_REGISTRY, c_transform)
        elif isinstance(c_transform, GeneralTransformation):
            return c_transform
        else:
            raise OutOfTypesException(c_transform, (str, dict, GeneralTransformation,))
    

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