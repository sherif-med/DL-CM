from dl_cm.utils.registery import Registry
import torchmetrics
from torchmetrics.metric import Metric
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.common import DLCM
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
from dl_cm.common.typing import namedEntitySchema

METRICS_REGISTRY = Registry("Metrics")

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
    if isinstance(attr, type) and issubclass(attr, torchmetrics.metric.Metric):
        METRICS_REGISTRY.register(attr)


