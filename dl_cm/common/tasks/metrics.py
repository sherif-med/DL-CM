from dl_cm.utils.registery import Registry
import torchmetrics
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.common import DLCM

METRICS_REGISTRY = Registry("Metrics")

class BaseMetric(torchmetrics.metric.Metric, DLCM):

    @staticmethod
    def registry() -> Registry:
        return METRICS_REGISTRY

class specificMetric(BaseMetric):
    
    def __init__(self, *args, **kwargs):
        self.predicted_map_key = kwargs.pop("predicted_map_key", None)
        self.target_map_key = kwargs.pop("target_map_key", None)
        super().__init__(*args, **kwargs)

    def update(self, preds, target):
        preds = preds.get(self.predicted_map_key) if self.predicted_map_key else preds
        target = preds.get(self.target_map_key) if self.target_map_key else target
        super().update(preds, target)

class MetricsFactory(BaseFactory):

    @staticmethod
    def base_class()-> type:
        return BaseMetric
    
# Register all torchmetrics metrics
for name in dir(torchmetrics):
    attr = getattr(torchmetrics, name)
    if isinstance(attr, type) and issubclass(attr, torchmetrics.metric.Metric):
        METRICS_REGISTRY.register(attr)


