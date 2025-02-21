from abc import ABC, abstractmethod

import pydantic as pd
import torch

from dl_cm.common import DLCM
from dl_cm.common.models import MODELS_REGISTERY
from dl_cm.utils.registery import Registry


class BaseModel(torch.nn.Module, DLCM, ABC):
    @staticmethod
    def registry() -> Registry:
        return MODELS_REGISTERY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
