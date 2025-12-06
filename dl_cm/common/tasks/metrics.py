import torchmetrics
from torchmetrics.metric import Metric
from typing import Callable

from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.ppattern.named_mixin import NamedInstanceMixin
from dl_cm.utils.registery import Registry
from dl_cm.common.functions import FunctionsFactory

METRICS_REGISTRY = Registry("Metrics")


def base_metric_adapter(metric_cls: type[Metric]):
    """Wrap a torchmetrics.Metric (even factory ones like F1Score) to work with dict inputs."""

    def wrapper(*args, **kwargs):
        # Step 1: Instantiate the metric (handles F1Score.__new__)
        preds_key = kwargs.pop("preds_key", None)
        target_key = kwargs.pop("target_key", None)
        
        preds_preprocessor = None
        if "preds_preprocessor" in kwargs:
            preds_preprocessor:Callable = FunctionsFactory.create(kwargs.pop("preds_preprocessor"))
        
        target_preprocessor = None
        if "target_preprocessor" in kwargs:
            target_preprocessor:Callable = FunctionsFactory.create(kwargs.pop("target_preprocessor"))
        
        metric_instance = metric_cls(*args, **kwargs)

        # Step 2: Create a dynamic subclass of the *instance’s class*
        class WrappedMetric(metric_instance.__class__, BaseMetric):

            def instance_name(self):
                if self._instance_count[metric_instance.__class__] == 0:
                    return metric_instance.__class__.__name__
                else:
                    self._instance_count[metric_instance.__class__] += 1
                    return (
                        f"{metric_instance.__class__.__name__}_{self._instance_count[metric_instance.__class__] - 1}"
                    )

            def update(self, preds: dict, target: dict = None):
                pred_tensor = preds[self.preds_key] if self.preds_key else preds
                pred_tensor = preds_preprocessor(pred_tensor) if preds_preprocessor else pred_tensor
                if target is None:
                    super().update(pred_tensor)
                    return
                target_tensor = target[self.target_key] if self.target_key else target
                target_tensor = target_preprocessor(target_tensor) if target_preprocessor else target_tensor
                super().update(pred_tensor, target_tensor)

        # Step 3: Create a new instance WITHOUT re-calling __init__
        wrapped = WrappedMetric.__new__(WrappedMetric)
        wrapped.__dict__ = metric_instance.__dict__.copy()  # copy state
        BaseMetric.__init__(wrapped, preds_key, target_key)  # initialize BaseMetric separately

        return wrapped

    return wrapper


class BaseMetric(NamedInstanceMixin, DLCM):
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
