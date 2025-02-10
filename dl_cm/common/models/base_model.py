import torch
from dl_cm.common import DLCM
from dl_cm.utils.registery import Registry
from dl_cm.common.models import MODELS_REGISTERY
import pydantic as pd

class BaseModel(torch.nn.Module, DLCM):

    @staticmethod
    def registry() -> Registry:
        return MODELS_REGISTERY
    
    def __init__(self, model_config:dict):
        super().__init__()
        self.model_config: dict = model_config
        #self.save_hyperparameters(model_config)
        #self.hparams["model_name"] = type(self).__name__
    
    def forward(self, x):
        raise NotImplementedError
    
    @classmethod
    def get_prediction_schema(cls) -> pd.BaseModel:
        """Returns the expected prediction schema of the model"""
        raise NotImplementedError
    

class IdModel(BaseModel):
    def forward(self, x):
        return x
    
    