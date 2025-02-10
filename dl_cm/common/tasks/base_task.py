import pytorch_lightning as pl
from dl_cm.common.tasks.optimizer import load_optimizer_from_config, load_lr_scheduler_from_config
import torch
from dl_cm.common import DLCM
from dl_cm.common.tasks import TASKS_REGISTERY
from typing import Dict
from dl_cm.common.typing import lossOutputStruct, Registry
from dataclasses import dataclass
import pydantic as pd
from dl_cm.common.typing import namedEntitySchema, Any
from dl_cm.utils.ppattern.data_validation import validationMixin

@dataclass
class StepOutputStruct:
    loss: lossOutputStruct | torch.Tensor
    predictions: Dict[str, Any] | torch.Tensor
    targets: Dict[str, Any] | torch.Tensor = None
    inputs: Dict | torch.Tensor = None
    metadata: Dict = None
    auxiliary: Dict = None

@dataclass
class StepInputStruct:
    inputs: Dict[str, Any] | torch.Tensor
    targets: Dict[str, Any] | torch.Tensor = None
    metadata: Dict = None
    auxiliary: Dict = None

class BaseTask(pl.LightningModule, DLCM, validationMixin):
    """
    Base class for all tasks in the DLCM framework.
    A task is a self-contained unit of work that encapsulates a specific learning problem.
    It is responsible for defining the models, optimizers, and learning logic for a particular problem through learner class.
    """

    @staticmethod
    def registry() -> Registry:
        return TASKS_REGISTERY

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            learner: namedEntitySchema = None
        return ValidConfig

    def __init__(self, task_config:dict):
        validationMixin.__init__(self, task_config)
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
