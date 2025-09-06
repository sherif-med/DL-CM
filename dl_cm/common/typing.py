from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar

import pydantic as pd
import torch

TT = TypeVar("TT")
OneOrMany: TypeAlias = TT | Iterable[TT]


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
    loss: torch.Tensor
    losses: lossOutputStruct
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
