from dl_cm.utils.registery import Registry
import torchmetrics
from torchmetrics.metric import Metric
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.common import DLCM
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
from dl_cm.common.typing import namedEntitySchema

METRICS_REGISTRY = Registry("Metrics")

def decorate_metric(metric_cls: type[Metric]):
    """Decorator to make any torchmetrics.Metric work with specific dictionary inputs keys"""
    class WrappedMetric(metric_cls, BaseMetric):
        def __init__(self, preds_key: str = None, target_key: str = None, *args, **kwargs):
            """Wraps a metric to extract values from dictionary inputs."""
            super().__init__(*args, **kwargs)
            BaseMetric.__init__(self)
            self.preds_key = preds_key
            self.target_key = target_key

        def update(self, preds: dict, target: dict=None):
            """Extract tensors and call the original update method."""
            pred_tensor = preds[self.preds_key] if self.preds_key else preds
            if target is None:
                super().update(pred_tensor)
                return
            target_tensor = target[self.target_key] if self.target_key else target
            super().update(pred_tensor, target_tensor)

    return WrappedMetric

class BaseMetric(Metric, DLCM, validationMixin):

    @staticmethod
    def registry() -> Registry:
        return METRICS_REGISTRY
    
    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(namedEntitySchema):
            preds_key: str = None
            target_key: str = None
        return ValidConfig
    
    def __init__(self, config:dict):
        validationMixin.__init__(self, config)

class MetricsFactory(BaseFactory):

    @staticmethod
    def base_class()-> type:
        return BaseMetric
    
# Register all torchmetrics metrics
for name in dir(torchmetrics):
    attr = getattr(torchmetrics, name)
    if isinstance(attr, type) and issubclass(attr, Metric):
        METRICS_REGISTRY.register(obj=decorate_metric(attr), name=attr.__name__)


