from typing import List, Tuple
from dl_cm.utils.registery import Registry
import torchmetrics
from dl_cm.utils.ppattern.factory import BaseFactory

METRICS_REGISTRY = Registry("Metrics")

class BaseMetric(torchmetrics.metric.Metric):
    pass

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


