import pytorch_lightning as pl
from dl_cm.common.tasks.optimizer import load_optimizer_from_config, load_lr_scheduler_from_config
import torch
from dl_cm.common import DLCM
from dl_cm.common.tasks import TASKS_REGISTERY
from typing import List, Dict
from dl_cm.common.typing import lossOutputStruct, Registry
from dataclasses import dataclass

@dataclass
class StepOutputStruct:
    loss: lossOutputStruct | torch.Tensor
    predictions: Dict | torch.Tensor
    target: Dict | torch.Tensor = None
    inputs: Dict | torch.Tensor = None
    metadata: Dict = None
    auxiliary: Dict = None

@dataclass
class StepInputStruct:
    inputs: Dict | torch.Tensor
    targets: Dict | torch.Tensor = None
    metadata: Dict = None
    auxiliary: Dict = None

class BaseTask(pl.LightningModule, DLCM):

    @staticmethod
    def registry() -> Registry:
        return TASKS_REGISTERY

    def __init__(self, task_config:dict):
        super().__init__()
        self.task_config: dict = task_config
        self.save_hyperparameters(task_config)
        self.hparams["task_name"] = type(self).__name__
        self.lr = task_config.get("optimizer").get("params").get("lr")
                
    def configure_optimizers(self):
        optimizer = load_optimizer_from_config(self.model.parameters(), self.task_config.get("optimizer"))
        lr_scheduler = load_lr_scheduler_from_config(optimizer, self.task_config.get("lr_scheduler")) if self.task_config.get("lr_scheduler") else None
        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]
