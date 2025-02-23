import pytorch_lightning as pl
import torch

from dl_cm.common import DLCM
from dl_cm.common.learning import LearnersFactory
from dl_cm.common.learning.optimizable_learner import OptimizableLearner
from dl_cm.common.typing import namedEntitySchema
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

TASKS_REGISTERY = Registry("Tasks")


class BaseTask(pl.LightningModule, DLCM):
    """
    Base class for all tasks in the DLCM framework.
    A task is a self-contained unit of work that encapsulates a specific learning problem.
    It is responsible for defining the models, optimizers, and learning logic for a particular problem through learner class.
    """

    @staticmethod
    def registry() -> Registry:
        return TASKS_REGISTERY

    def __init__(
        self, learner: OptimizableLearner | namedEntitySchema, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.learner: OptimizableLearner = LearnersFactory.create(learner)
        self.hparams["task_name"] = type(self).__name__
        self.save_hyperparameters(kwargs)

    def configure_optimizers(self):
        optimizer = self.learner.optimizers
        lr_scheduler = self.learner.lr_schedulers
        return [optimizer], [lr_scheduler]

    @staticmethod
    def load_from_checkpoint(ckpt_path: str, **kwargs) -> "BaseTask":
        ckpt = torch.load(ckpt_path, map_location="cpu")
        task_name = ckpt["hyper_parameters"]["name"]
        task_class: pl.LightningModule = TASKS_REGISTERY.get(task_name)
        del ckpt
        loaded_task = task_class.load_from_checkpoint(ckpt_path, **kwargs)
        return loaded_task


class TasksFactory(BaseFactory[BaseTask]):
    @staticmethod
    def base_class() -> type[BaseTask]:
        return BaseTask
