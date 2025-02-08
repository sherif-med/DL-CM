from dl_cm.utils.registery import Registry
from .base_model import BaseModel, IdModel
from dl_cm.utils.ppattern.factory import BaseFactory

MODELS_REGISTERY = Registry("Models")

class ModelsFactory(BaseFactory):

    @staticmethod
    def base_class()-> type:
        return BaseModel
    
    @classmethod
    def default_instance(cls)-> BaseModel:
        return IdModel()
    
