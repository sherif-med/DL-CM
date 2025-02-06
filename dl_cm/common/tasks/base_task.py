import pytorch_lightning as pl
from dl_cm.common.tasks.optimizer import load_optimizer_from_config, load_lr_scheduler_from_config
import torch

from typing import List, Dict
from typing import TypedDict

class StepOutputDict(TypedDict):
    loss: Dict | torch.Tensor
    prediction: Dict | torch.Tensor
    target: Dict | torch.Tensor
    input: Dict | torch.Tensor
    metadata: Dict
    auxiliary: Dict

class BaseTask(pl.LightningModule):
    def __init__(self, task_config:dict):
        super().__init__()
        self.task_config: dict = task_config
        self.save_hyperparameters(task_config)
        self.hparams["task_name"] = type(self).__name__
        self.lr = task_config.get("optimizer").get("params").get("lr")
        
    def info(self):
        """
        Returns the task description defined in the child class.
        """
        # Check if the child class has a DESCRIPTION attribute
        if hasattr(self, 'DESCRIPTION'):
            return self.DESCRIPTION
        else:
            # Return a default message or raise an error if DESCRIPTION is not defined
            return "No description provided."
    
    def step(self, batch, compute_loss=True) -> StepOutputDict:
        raise NotImplementedError
        
    def configure_optimizers(self):
        optimizer = load_optimizer_from_config(self.model.parameters(), self.task_config.get("optimizer"))
        lr_scheduler = load_lr_scheduler_from_config(optimizer, self.task_config.get("lr_scheduler")) if self.task_config.get("lr_scheduler") else None
        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]
