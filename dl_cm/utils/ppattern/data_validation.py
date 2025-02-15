from pydantic import BaseModel
from dl_cm import _logger as logger
from dl_cm.utils.ppattern.init_check_mixin import InitCheckMixin
from abc import ABC, abstractmethod

class validationMixin(InitCheckMixin, ABC):

    def __init__(self, data):
        InitCheckMixin.__init__(self)
        self.validate(data)

    def validate(self, data)-> None:
        try:
            _ = type(self).config_schema()(data) # TODO this should be fixed by making config_schema a classmethod
        except ValueError as e:
            logger.error("Data validation failed for class %s", self.__class__.__name__)
            logger.error(e)
            exit(1)

    @abstractmethod
    @staticmethod
    def config_schema()-> BaseModel:
        pass
