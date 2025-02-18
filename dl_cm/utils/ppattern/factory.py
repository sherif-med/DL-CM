"""
Module containing the BaseFactory class, which provides a base class for creating factories in the
dl_cm framework. A factory is responsible for creating instances of a particular class, and this
module provides a generic way to implement factories. The BaseFactory class provides a way to
create instances of a class from a variety of input parameters, including strings, dictionaries,
and instances of the class itself.
"""

import collections
import collections.abc
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from dl_cm.config_loaders import load_named_entity
from dl_cm.utils.exceptions import OutOfTypesException
from dl_cm.utils.registery import Registry, registeredClassMixin

T = TypeVar("T", bound=registeredClassMixin)


class BaseFactory(Generic[T], ABC):
    """
    A base class for creating factories.
    The BaseFactory class has the following methods:
        - registry: Returns the registry of classes associated with the factory.
        - base_class: Returns the base class of the factory.
        - default_instance: Returns a default instance of the class.
        - create: Creates an instance of the class from a variety of input parameters.
    """

    @classmethod
    def registry(cls) -> Registry:
        return cls.base_class().registry()

    @staticmethod
    @abstractmethod
    def base_class() -> T:
        """
        Return the base class associated with the factory.
        This method should be overridden by subclasses to return the specific base class
        that the factory is responsible for creating instances of.
        """

    @classmethod
    @abstractmethod
    def default_instance(cls) -> T:
        """
        Return a default instance of the class associated with the factory.
        This method should be overridden by subclasses to provide a default instance
        of the specific class that the factory is responsible for creating.
        """

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

        if param is None:
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
            raise OutOfTypesException(
                param,
                (
                    str,
                    dict,
                    cls.base_class(),
                ),
            )
