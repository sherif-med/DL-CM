import torch
from dl_cm.common import DLCM
from dl_cm.utils.registery import Registry
from dl_cm.common.models import MODELS_REGISTERY
import pydantic as pd
from abc import ABC, abstractmethod

class BaseModel(torch.nn.Module, DLCM, ABC):

    @staticmethod
    def registry() -> Registry:
        return MODELS_REGISTERY
    
    def __init__(self, model_config:dict):
        super().__init__()
        self.model_config: dict = model_config
        #self.save_hyperparameters(model_config)
        #self.hparams["model_name"] = type(self).__name__
    
    @abstractmethod
    def forward(self, x):
        pass
    
    @classmethod
    @abstractmethod
    def get_prediction_schema(cls) -> pd.BaseModel:
        """Returns the expected prediction schema of the model"""
        pass
    

class IdModel(BaseModel):
    def forward(self, x):
        return x
    
    