
from typing import Any
import torch
from dataclasses import dataclass
import pydantic as pd

class namedEntitySchema(pd.BaseModel):
    name: str
    params: dict = {}

@dataclass
class lossOutputStruct:
    name: str
    losses: dict[str, float | torch.Tensor]

    def value(self):
        return self.losses[self.name]

@dataclass
class StepOutputStruct:
    loss: lossOutputStruct | torch.Tensor
    predictions: dict[str, Any] | torch.Tensor
    targets: dict[str, Any] | torch.Tensor = None
    inputs: dict | torch.Tensor = None
    metadata: dict = None
    auxiliary: dict = None

@dataclass
class StepInputStruct:
    inputs: dict[str, Any] | torch.Tensor
    targets: dict[str, Any] | torch.Tensor = None
    metadata: dict = None
    auxiliary: dict = None