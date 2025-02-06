from dl_cm.utils.registery import Registry
from torchmetrics import MeanMetric
import torch.nn as nn
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm import _logger as logger
from torchmetrics import MetricCollection


class BaseLoss(nn.modules.loss._Loss):
    def __init__(self):
        super(BaseLoss, self).__init__()
    
    def as_metric_collection(self) -> MetricCollection:
        return MetricCollection({self.__class__.__name__: MeanMetric()})

    def forward(self, prediction, target):
        """
        Forward pass of the loss.
        :return: a dictionary with the key being the name of the loss and the value being the loss
        """
        return {self.__class__.__name__: super().forward(prediction, target)}


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
        self.losses = nn.ModuleList(losses)
        
        if weights is None:
            logger.info("Defaulting to equal weighting of losses")
            weights = [1.0] * len(losses)
        if len(weights) != len(losses) or any({w is None for w in weights}):
            logger.warning("Number of weights does not match number of losses. Defaulting to equal weighting of losses")
            weights = [1.0] * len(losses)
        
        self.weights = weights
        if sum(self.weights)!=1:
            self.weights =list(map(lambda x:x/sum(self.weights), self.weights))

    def forward(self, prediction, target):
        losses_dict = {}
        total_loss = 0.0
        # Iterate over each loss function and corresponding weight
        for loss_fn, weight in zip(self.losses, self.weights):
            c_loss_value = loss_fn(prediction, target)
            losses_dict[loss_fn.__class__.__name__] = c_loss_value
            total_loss += c_loss_value * weight
        losses_dict[self.__class__.__name__] = total_loss
        return losses_dict
    
    def as_metric_collection(self):
        # Adding current loss to the metric collection
        metric_collection = MetricCollection({self.__class__.__name__: MeanMetric()})
        # Adding child losses to the metric collection
        loss_fn : BaseLoss
        for loss_fn in self.losses:
            metric_collection.add_metrics(loss_fn.as_metric_collection())
        return metric_collection

CRITIREON_REGISTRY = Registry("Critireon")
CRITIREON_REGISTRY.register(CombinedLoss)

class CritireonFactory(BaseFactory):
    
    @classmethod
    def base_class(cls)-> type:
        return BaseLoss
    
    @classmethod
    def registry(cls) -> Registry:
        return CRITIREON_REGISTRY

for name in dir(nn.modules.loss):
    attr = getattr(nn.modules.loss, name)
    if isinstance(attr, type) and issubclass(attr, nn.modules.loss._Loss):
        CRITIREON_REGISTRY.register(attr)

from segmentation_models_pytorch.losses import DiceLoss
CRITIREON_REGISTRY.register(DiceLoss)
