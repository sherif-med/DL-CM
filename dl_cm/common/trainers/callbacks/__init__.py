from dl_cm.utils.registery import Registry

CALLBACKS_REGISTERY = Registry("Callbacks")

from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

CALLBACKS_REGISTERY.register(TQDMProgressBar)
CALLBACKS_REGISTERY.register(ModelCheckpoint)
