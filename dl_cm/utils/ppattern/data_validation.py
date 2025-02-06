from pydantic import BaseModel
from dl_cm import _logger as logger

class validationMixin:

    def __init__(self, data):
        self.validate(data)
    
    def validate(self, data)-> None:
        try:
            parsed = self.config_schema()(data)
        except ValueError as e:
            logger.error("Data validation failed for class %s", self.__class__.__name__)
            logger.error(e)
            exit(1)
    
    @staticmethod
    def config_schema(cls)-> BaseModel:
        raise NotImplementedError
