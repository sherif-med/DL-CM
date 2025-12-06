from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

LOGGERS_REGISTERY = Registry("Loggers")


class BaseLogger(DLCM):
    @staticmethod
    def registry() -> Registry:
        return LOGGERS_REGISTERY

    def __init__(self, *args, **config) -> None:
        super().__init__(*args, **config)


class LoggersFactory(BaseFactory[BaseLogger]):
    @staticmethod
    def base_class() -> type[BaseLogger]:
        return BaseLogger


import pytorch_lightning.loggers as pl_loggers

for name in dir(pl_loggers):
    attr = getattr(pl_loggers, name)
    if isinstance(attr, type) and issubclass(attr, pl_loggers.Logger):
        _ = DLCM.base_class_adapter(attr, base_cls=BaseLogger)


from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import wandb
import torch, numpy
from typing import Union, List

def log_image_all_loggers(
    trainer, 
    tag: str, 
    image: Union[torch.Tensor, numpy.ndarray], 
    step: int = None
):
    """Log image to all configured loggers"""
    step = step if step is not None else trainer.current_epoch
    
    # Handle both single logger and list of loggers
    loggers = trainer.loggers if isinstance(trainer.loggers, list) else [trainer.logger]
    # Convert to numpy if needed
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    else:
        image_np = image
    
    for logger in loggers:
        try:
            if isinstance(logger, TensorBoardLogger):
                # Determine format from shape
                dataformats = "HWC" if image_np.ndim == 3 else "HW"
                logger.experiment.add_image(tag, image_np, step, dataformats=dataformats)
            
            elif isinstance(logger, WandbLogger):
                logger.experiment.log({
                    tag: wandb.Image(image_np),
                    "epoch": trainer.current_epoch,
                    #"trainer/global_step": trainer.global_step
                })
        except Exception as e:
            print(f"Warning: Failed to log image to {type(logger).__name__}: {e}")
