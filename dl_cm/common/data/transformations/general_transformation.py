from abc import ABC, abstractmethod

from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

from . import TRANSFORMATION_REGISTRY


class GeneralTransformation(DLCM):
    """
    A class representing a general transformation.

    This class is designed to apply a forward transformation function to an item.
    It can be extended or used as a base class for specific transformations like preprocessing.

    Attributes:
        __fwd__fn__: A callable function that takes items and returns transformed items.

    Methods:
        __call__(item, **kwargs): Applies the forward transformation function to the given item.
        registry(): Returns the registry associated with this transformation type.
    """

    def __init__(self, fwdfn):
        """
        Initializes a GeneralTransformation.

        Args:
            fwdfn: A function that takes some items and returns transformed items.
        """
        self.__fwd__fn__ = fwdfn

    def __fwd__(self, *args):
        return self.__fwd__fn__(*args)

    def __call__(self, item, **kwargs):
        return self.__fwd__(item, **kwargs)

    @staticmethod
    def registry() -> Registry:
        return TRANSFORMATION_REGISTRY


class GeneralTransformationFactory(BaseFactory[GeneralTransformation]):
    @staticmethod
    def base_class(similar=False) -> type[GeneralTransformation]:
        return GeneralTransformation


class GeneralIrrevirsibleTransformation(GeneralTransformation, ABC):
    def __call__(self, item, **kwargs):
        return self.__fwd__(item, **kwargs)

    @abstractmethod
    def __fwd__(self, *args):
        pass


class GeneralRevrsibleTransformation(GeneralTransformation):
    def __init__(self, fwdfn, rwdfn):
        self.__fwd__fn__ = fwdfn
        self.__rwd__fn__ = rwdfn

    def __invert__(self):
        return GeneralRevrsibleTransformation(self.__rwd__fn__, self.__fwd__fn__)

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


class SupervisionTransformation:
    """
    Applies transformation on Inputs and Target keys
    """

    def __init__(self):
        self._supervision_keys = ("inputs", "targets")

    def __call__(self, item, **kwargs):
        out_item = {}
        for s_k in self._supervision_keys:
            if s_k in item:
                out_item[s_k] = super().__call__(item[s_k], **kwargs)
        return out_item