from abc import ABC, abstractmethod

import pydantic as pd
import torch

from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

MODELS_REGISTERY = Registry("Models")


class BaseModel(torch.nn.Module, DLCM, ABC):
    @staticmethod
    def registry() -> Registry:
        return MODELS_REGISTERY

    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        DLCM.__init__(self, *args, **kwargs)

    @abstractmethod
    def forward(self, x):
        pass

    @classmethod
    # @abstractmethod
    def get_prediction_schema(cls) -> pd.BaseModel:
        """Returns the expected prediction schema of the model"""
        raise NotImplementedError


class IdModel(BaseModel):
    def forward(self, x):
        return x


class ModelsFactory(BaseFactory[BaseModel]):
    @staticmethod
    def base_class(similar=False) -> type[BaseModel]:
        if similar:
            return (BaseModel,)
        return BaseModel

    @classmethod
    def default_instance(cls) -> BaseModel:
        return IdModel()


from . import samples

__all__ = ["samples", "BaseModel", "IdModel", "ModelsFactory"]
