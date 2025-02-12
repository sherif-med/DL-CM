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
from dl_cm.utils.exceptions import OutOfTypesException
from dl_cm.utils.ppattern.data_validation import validationMixin

@dataclass
class lossOutputStruct:
    name: str
    losses: dict[str, float | torch.Tensor]

    def value(self):
        return self.losses[self.name]

CRITIREON_REGISTRY = Registry("Critireon")

def base_loss_adapter(loss_cls: type[nn.modules.loss._Loss]):
    """Decorator to make any external loss inherit from BaseLoss
       and adapt output to lossOutputStruct
    """
    class WrappedLoss(loss_cls, BaseLoss):
        def __init__(self, config:dict):
            """Wraps a loss to extract values from dictionary inputs."""
            BaseLoss.__init__(self, config)
            super().__init__(config.get("params", {}))
        
        @staticmethod
        def adapt_output_struct(output_loss):
            if isinstance(output_loss, lossOutputStruct):
                return output_loss
            if isinstance(output_loss, (torch.Tensor, float)):
                output_loss = lossOutputStruct(name=loss_cls.__name__, losses={loss_cls.__name__: output_loss})
                return output_loss
            if isinstance(output_loss, dict):
                if not loss_cls.__name__ in output_loss:
                    logger.critical(f"Metric {loss_cls.__name__} is not in loss dict!")
                output_loss = lossOutputStruct(name=loss_cls.__name__, losses=output_loss)
                return output_loss
            raise OutOfTypesException(output_loss, (lossOutputStruct, torch.Tensor, float, dict))

        def forward(self, preds: dict, target: dict=None):
            """Extract tensors and call the original update method."""
            pred_tensor = preds[self.preds_key] if self.preds_key else preds
            if target is None:
                loss_output = super().forward(pred_tensor)
                return self.adapt_output_struct(loss_output)
            target_tensor = target[self.target_key] if self.target_key else target
            loss_output = super().forward(pred_tensor, target_tensor)
            return self.adapt_output_struct(loss_output)

    return WrappedLoss

class BaseLoss(NamedMixin, DLCM, validationMixin):

    @staticmethod
    def registry() -> Registry:
        return CRITIREON_REGISTRY
    
    def config_schema()-> pd.BaseModel:
        class ValidConfig(namedEntitySchema):
            preds_key: str = None
            target_key: str = None
        return ValidConfig
    
    def __init__(self, config:dict):
        NamedMixin.__init__(self)
        validationMixin.__init__(self, config)
        self.preds_key = config.get("preds_key")
        self.target_key = config.get("target_key")
    
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

for name in dir(nn.modules.loss):
    attr = getattr(nn.modules.loss, name)
    if isinstance(attr, type) and issubclass(attr, nn.modules.loss._Loss):
        CRITIREON_REGISTRY.register(obj=attr, name=name, base_class_adapter=base_loss_adapter)

from segmentation_models_pytorch.losses import DiceLoss
CRITIREON_REGISTRY.register(obj=DiceLoss, name="DiceLoss", base_class_adapter=base_loss_adapter)
