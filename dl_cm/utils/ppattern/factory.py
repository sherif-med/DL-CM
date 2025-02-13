import collections.abc
from dl_cm.utils.registery import Registry
from dl_cm.config_loaders import load_named_entity
from dl_cm.utils.exceptions import OutOfTypesException
import collections
from typing import TypeVar, Generic
import pydantic as pd
from dl_cm.utils.registery import registeredClassMixin
from abc import ABC, abstractmethod

class namedEntitySchema(pd.BaseModel):
    name: str
    params: dict = {}

T = TypeVar("T", bound=registeredClassMixin)
class BaseFactory(Generic[T], ABC):

    @classmethod
    def registry(cls) -> Registry:
        return cls.base_class().registry()
    
    @staticmethod
    @abstractmethod
    def base_class() -> T:
        pass
    
    @classmethod
    @abstractmethod
    def default_instance(cls) -> T:
        pass

    @classmethod
    def create(cls, param: str | dict | T | collections.abc.Iterable[T]) -> T | list[T]:
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
