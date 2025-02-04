from dl_cm.utils.registery import Registry
import torchmetrics
from dl_cm.utils.ppattern.factory import BaseFactory

METRICS_REGISTRY = Registry("Metrics")

class BaseMetric(torchmetrics.metric.Metric):
    pass

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

    @classmethod
    def base_class(cls)-> type:
        return BaseMetric
    
    @classmethod
    def registry(cls) -> Registry:
        return METRICS_REGISTRY

# Register all torchmetrics metrics
for name in dir(torchmetrics):
    attr = getattr(torchmetrics, name)
    if isinstance(attr, type) and issubclass(attr, torchmetrics.metric.Metric):
        METRICS_REGISTRY.register(attr)


