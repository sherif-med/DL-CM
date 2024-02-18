from typing import Tuple
from dl_cm.utils.registery import Registry
from torchmetrics import MeanMetric
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super(CombinedLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        # If weights are not provided, default to equal weighting
        if weights is None or any({w is None for w in weights}):
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
        losses_dict["total_loss"] = total_loss
        return losses_dict

def load_critireon_from_config(critireon_config)->CombinedLoss:
    
    critireon_def = critireon_config.get("def")    
    sub_losses = []
    sub_weight = []    
    for c_loss_def in critireon_def:
        sub_losses.append(CRITIREON_REGISTRY.get(c_loss_def.get("name"))(**c_loss_def.get("params")))
        sub_weight.append(c_loss_def.get("weight"))
    
    return CombinedLoss(sub_losses, sub_weight)

def init_loss_metrics_dict_from_config(critireon_config)->Tuple[nn.ModuleDict, dict]:
    """
    Returns a tuple composed of 1) ModuleDict for loss metric tracking 2) dict for logging behavior of loss metrics
    """
    
    total_log_on_epoch = critireon_config.get("log_on_epoch")
    total_log_on_step = critireon_config.get("log_on_step")
    
    metrics_log_flags = {
        "total_loss":{
            "log_on_epoch" : total_log_on_epoch,
            "log_on_step" : total_log_on_step
        }
    }
    
    metrics_elements = {
        "total_loss": MeanMetric()
    }
    
    critireon_def = critireon_config.get("def") 
    
    for c_loss_def in critireon_def:
        metrics_log_flags[c_loss_def.get("name")] = {
            "log_on_epoch":c_loss_def.get("log_on_epoch", True),
            "log_on_step":c_loss_def.get("log_on_step", False)
        }
        metrics_elements[c_loss_def.get("name")] = MeanMetric()
    
    loss_metrics = nn.ModuleDict(metrics_elements)
    return loss_metrics, metrics_log_flags
        
    

CRITIREON_REGISTRY = Registry("Critireon")

for name in dir(nn.modules.loss):
    attr = getattr(nn.modules.loss, name)
    if isinstance(attr, type) and issubclass(attr, nn.modules.loss._Loss):
        CRITIREON_REGISTRY.register(attr)

from segmentation_models_pytorch.losses import DiceLoss
CRITIREON_REGISTRY.register(DiceLoss)

