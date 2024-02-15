from dl_cm.utils.registery import Registry

LOGGERS_REGISTERY = Registry("Loggers")

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
LOGGERS_REGISTERY.register(TensorBoardLogger)