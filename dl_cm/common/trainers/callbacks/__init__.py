from dl_cm.utils.registery import Registry
from dl_cm.common import DLCM
CALLBACKS_REGISTERY = Registry("Callbacks")

class baseCallback(DLCM):
    
    @staticmethod
    def registry() -> Registry:
        return CALLBACKS_REGISTERY

from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.callbacks import LearningRateMonitor

CALLBACKS_REGISTERY.register(TQDMProgressBar)
CALLBACKS_REGISTERY.register(ModelCheckpoint)
CALLBACKS_REGISTERY.register(LearningRateFinder)
CALLBACKS_REGISTERY.register(LearningRateMonitor)
