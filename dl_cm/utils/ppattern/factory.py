from dl_cm.utils.registery import Registry
from dl_cm.config_loaders import load_named_entity
from dl_cm.utils.exceptions import OutOfTypesException
import collections
from typing import Self
from typing import TypeVar, Type
from dl_cm import _logger as logger
from dl_cm.common import DLCM

class BaseFactory:

    @classmethod
    def registry(cls) -> Registry:
        raise cls.base_class().registry()
    
    @staticmethod
    def base_class() -> Type["DLCM"]:
        raise NotImplementedError
    
    @classmethod
    def default_instance(cls) -> DLCM:
        logger.critical("Default instance not implemented for factory %s", cls.__name__)
        raise NotImplementedError

    @classmethod
    def create(cls, param: str | dict | Type["DLCM"] | collections.abc.Iterable[Type["DLCM"]]) -> "DLCM" | list["DLCM"]:
        """
        Create an instance of the class, or a list of instances from the parameters.

        Parameters
        ----------
        param : str, dict, cls.base_class(), collections.abc.Iterable
            Parameters to create the instance. If str, it is the name of the class to create.
            If dict, it is the configuration for the class to create.
            If cls.base_class(), it is the instance to return.
            If collections.abc.Iterable, it is a list of parameters to create a list of instances.

        Returns
        -------
        Self
            An instance of the class, or a list of instances.

        Raises
        ------
        OutOfTypesException
            If the param is not of the expected types.
        """

        if not param:
            return cls.default_instance()
        if isinstance(param, str):
            dataset_class = cls.registry().get(param)
            return dataset_class()
        elif isinstance(param, dict):
            return load_named_entity(cls.registry(), param)
        elif isinstance(param, cls.base_class()):
            return param
        elif isinstance(param, collections.abc.Iterable):
            return type(param)(cls.create(p) for p in param)
        else:
            raise OutOfTypesException(param, (str, dict, cls.base_class(),))
