from typing import List
import pytorch_lightning as pl
from dl_cm.common.trainers.callbacks import CALLBACKS_REGISTERY
from dl_cm.common.trainers.loggers import LOGGERS_REGISTERY


def callbacks_from_config(trainer_config:dict) -> List[pl.Callback]:
    """
    Returns a list of initialized callbacks from config
    """    
    loaded_callbacks = []
    
    for c_callback_def in trainer_config.get("callbacks", {}):
        callback_cls = CALLBACKS_REGISTERY.get(c_callback_def.get("name"))
        callback_params = c_callback_def.get("params")
        loaded_callbacks.append(
            callback_cls(**callback_params)
        )
    return loaded_callbacks

def loggers_from_config(trainer_config:dict):
    """
    Returns a list of initialized loggers from config
    """
    loaded_loggers = []
    
    for c_logger_def in trainer_config.get("loggers", {}):
        logger_cls = LOGGERS_REGISTERY.get(c_logger_def.get("name"))
        logger_params = c_logger_def.get("params")
        loaded_loggers.append(
            logger_cls(**logger_params)
        )
    return loaded_loggers

def trainer_from_config(trainer_config:dict) -> pl.Trainer:
    """
    Returns a pre configured trainer using config
    """    
    callbacks = callbacks_from_config(trainer_config)
    loggers = loggers_from_config(trainer_config)    
    kwargs = trainer_config.get("params", {}) | { "callbacks": callbacks, "logger": loggers }    
    trainer = pl.Trainer(**kwargs)
    return trainer