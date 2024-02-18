from typing import List, Tuple
from dl_cm.utils.registery import Registry
from torchmetrics import MetricCollection
import torchmetrics

METRICS_REGISTRY = Registry("Metrics")

for name in dir(torchmetrics):
    attr = getattr(torchmetrics, name)
    if isinstance(attr, type) and issubclass(attr, torchmetrics.metric.Metric):
        METRICS_REGISTRY.register(attr)

def init_metric_collection_from_config(metrics_config)->Tuple[MetricCollection]:
    """"""
    
    def init_mode_config(metrics_defs: List[dict], **kwargs)->MetricCollection:
        """"""
        metrics = []
        for c_metric_def in metrics_defs:
            c_metric_cls = METRICS_REGISTRY.get(c_metric_def.get("name"))
            metrics.append(
                c_metric_cls( **c_metric_def.get("params") )
            )
        metric_collection = MetricCollection(metrics, **kwargs)
        return metric_collection
    
    train_mode_config = metrics_config.get("train_mode")
    valid_mode_config = metrics_config.get("valid_mode")
    
    return init_mode_config(train_mode_config.get("defs"), prefix="train_"), init_mode_config(valid_mode_config.get("defs"), prefix="valid_")
