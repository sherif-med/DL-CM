import torchmetrics
from torchmetrics.metric import Metric

from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

METRICS_REGISTRY = Registry("Metrics")


def base_metric_adapter(metric_cls: type[Metric]):
    """Decorator to make any torchmetrics.Metric work with specific dictionary inputs keys"""

    class WrappedMetric(metric_cls, BaseMetric):
        def __init__(self, *args, **kwargs):
            """Wraps a metric to extract values from dictionary inputs."""
            super().__init__(*args, **kwargs.pop("params", {}))
            BaseMetric.__init__(self, *args, **kwargs)

        def update(self, preds: dict, target: dict = None):
            """Extract tensors and call the original update method."""
            pred_tensor = preds[self.preds_key] if self.preds_key else preds
            if target is None:
                super().update(pred_tensor)
                return
            target_tensor = target[self.target_key] if self.target_key else target
            super().update(pred_tensor, target_tensor)

    return WrappedMetric


class BaseMetric(DLCM):
    @staticmethod
    def registry() -> Registry:
        return METRICS_REGISTRY

    def __init__(self, preds_key: str = None, target_key: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds_key = preds_key
        self.target_key = target_key


class MetricsFactory(BaseFactory[BaseMetric]):
    @staticmethod
    def base_class() -> type[BaseMetric]:
        return BaseMetric


# Register all torchmetrics metrics
for name in dir(torchmetrics):
    attr = getattr(torchmetrics, name)
    if isinstance(attr, type) and issubclass(attr, Metric):
        METRICS_REGISTRY.register(
            obj=attr, name=attr.__name__, base_class_adapter=base_metric_adapter
        )
