from dl_cm.utils.registery import Registry
import torch
import pytorch_lightning as pl
from .base_task import BaseTask

TASKS_REGISTERY = Registry("Tasks")



def load_task_from_config(task_config:dict)->BaseTask:
    """
    Returns a predefined task loaded using config
    """    
    task_cls = TASKS_REGISTERY.get(task_config.get("name"))
    loaded_task = task_cls(task_config)
    return loaded_task


def load_task_from_checkpoint(ckpt_path:str)->BaseTask:
    """"""
    ckpt = torch.load(ckpt_path)
    task_name = ckpt["hyper_parameters"]["name"]
    task_class:pl.LightningModule = TASKS_REGISTERY.get(task_name)
    del ckpt
    loaded_task = task_class.load_from_checkpoint(ckpt_path)
    return loaded_task
