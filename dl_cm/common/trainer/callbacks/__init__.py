import pytorch_lightning as pl

from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

CALLBACKS_REGISTERY = Registry("Callbacks")


class baseCallback(pl.Callback, DLCM):
    @staticmethod
    def registry() -> Registry:
        return CALLBACKS_REGISTERY

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class CallbacksFactory(BaseFactory[baseCallback]):
    @staticmethod
    def base_class(similar=False) -> type[baseCallback]:
        return baseCallback


from pytorch_lightning.callbacks import LearningRateFinder, LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import BasePredictionWriter

_ = DLCM.base_class_adapter(TQDMProgressBar, base_cls=baseCallback)
_ = DLCM.base_class_adapter(ModelCheckpoint, base_cls=baseCallback)
_ = DLCM.base_class_adapter(LearningRateFinder, base_cls=baseCallback)
_ = DLCM.base_class_adapter(LearningRateMonitor, base_cls=baseCallback)
_ = DLCM.base_class_adapter(BasePredictionWriter, base_cls=baseCallback)

from .metric_logging_callback import MetricsLoggingCallback
from .metric_track_callback import metricTrackCallback
from .prediction_writer import ImagesPredictionWriter, PostPredictionCallback

__all__ = [
    "baseCallback",
    "CallbacksFactory",
    "MetricsLoggingCallback",
    "metricTrackCallback",
    "ImagesPredictionWriter",
    "PostPredictionCallback"
]
