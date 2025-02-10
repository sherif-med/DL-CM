from dl_cm.utils.registery import Registry
from torchmetrics import MeanMetric
import torch.nn as nn
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm import _logger as logger
from torchmetrics import MetricCollection
from dl_cm.utils.ppattern.named_mixin import NamedMixin
from dl_cm.common import DLCM
from dataclasses import dataclass
from dl_cm.common.typing import namedEntitySchema
import pydantic as pd
import torch

@dataclass
class lossOutputStruct:
    name: str
    losses: dict[str, float | torch.Tensor]

    def value(self):
        return self.losses[self.name]

CRITIREON_REGISTRY = Registry("Critireon")

def decorate_loss(metric_cls: type[nn.modules.loss._Loss]):
    """Decorator to make any nn.modules.loss._Loss work with specific dictionary inputs keys"""
    class WrappedLoss(metric_cls, BaseLoss):
        def __init__(self, preds_key: str = None, target_key: str = None, *args, **kwargs):
            """Wraps a loss to extract values from dictionary inputs."""
            super().__init__(*args, **kwargs)
            BaseLoss.__init__(self)
            self.preds_key = preds_key
            self.target_key = target_key

        def forward(self, preds: dict, target: dict=None):
            """Extract tensors and call the original update method."""
            pred_tensor = preds[self.preds_key] if self.preds_key else preds
            if target is None:
                super().forward(pred_tensor)
                return
            target_tensor = target[self.target_key] if self.target_key else target
            super().forward(pred_tensor, target_tensor)

    return WrappedLoss

class BaseLoss(nn.modules.loss._Loss, NamedMixin, DLCM):

    @staticmethod
    def registry() -> Registry:
        return CRITIREON_REGISTRY
    
    def config_schema()-> pd.BaseModel:
        class ValidConfig(namedEntitySchema):
            preds_key: str = None
            target_key: str = None
        return ValidConfig
    
    def __init__(self):
        NamedMixin.__init__(self)
    
    def as_metric_collection(self) -> MetricCollection:
        return MetricCollection({self.name(): MeanMetric()})

class CombinedLoss(BaseLoss):
    def __init__(self, losses: list[str, dict, BaseLoss], weights: list[float]=None):        
        """
        :param losses: A list of losses to be combined. Each loss can be either a string
            (in which case it is a key in the CRITIREON_REGISTRY), a dictionary (in which
            case it is passed to the CritireonFactory to construct a loss), or an instance
            of nn.modules.loss._Loss.
        :param weights: A list of weights to be used in the combination. If None, the
            weights will be set to equal values (i.e. the losses will be equally weighted).
            If the length of the weights list is not equal to the length of the losses list,
            the weights will be set to equal values.
        :return:
        """

        super(CombinedLoss, self).__init__()
        losses = CritireonFactory.create(critireon_config=losses)
        self.losses : list[BaseLoss] = nn.ModuleList(losses)
        
        if weights is None:
            logger.info("Defaulting to equal weighting of losses")
            weights = [1.0] * len(losses)
        if len(weights) != len(losses) or any({w is None for w in weights}):
            logger.warning("Number of weights does not match number of losses. Defaulting to equal weighting of losses")
            weights = [1.0] * len(losses)
        
        self.weights = weights
        if sum(self.weights)!=1:
            self.weights =list(map(lambda x:x/sum(self.weights), self.weights))

    def forward(self, prediction, target) -> lossOutputStruct:
        losses_dict = {}
        total_loss = 0.0
        # Iterate over each loss function and corresponding weight
        for loss_fn, weight in zip(self.losses, self.weights):
            c_loss_value = loss_fn(prediction, target)
            losses_dict[loss_fn.name()] = c_loss_value
            total_loss += c_loss_value * weight
        losses_dict[self.name()] = total_loss
        return lossOutputStruct(name=self.name(), losses=losses_dict)
    
    def as_metric_collection(self):
        # Adding current loss to the metric collection
        metric_collection = MetricCollection({self.name(): MeanMetric()})
        # Adding child losses to the metric collection
        loss_fn : BaseLoss
        for loss_fn in self.losses:
            metric_collection.add_metrics(loss_fn.as_metric_collection())
        return metric_collection

class CritireonFactory(BaseFactory):
    
    @staticmethod
    def base_class()-> type:
        return BaseLoss

import functools

def adapt_external_loss(cls):
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        output = cls(*args, **kwargs)
        if not isinstance(output, lossOutputStruct):
            # Assume output is a scalar loss value
            loss_value = output
            # Create a LossOutputStruct instance with the loss value
            adapted_output = lossOutputStruct(name=cls.__name__, losses={cls.__name__: loss_value})
            return adapted_output
        return output
    return wrapper    

for name in dir(nn.modules.loss):
    attr = getattr(nn.modules.loss, name)
    if isinstance(attr, type) and issubclass(attr, nn.modules.loss._Loss):
        CRITIREON_REGISTRY.register(obj=decorate_loss(adapt_external_loss(DiceLoss)), name=name)

from segmentation_models_pytorch.losses import DiceLoss
CRITIREON_REGISTRY.register(obj=decorate_loss(adapt_external_loss(DiceLoss)), name="DiceLoss")
