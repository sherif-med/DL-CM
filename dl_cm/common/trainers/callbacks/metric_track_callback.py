import pytorch_lightning as pl
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
from typing import List
from torchmetrics import MetricCollection
from dl_cm.common.tasks.metrics import MetricsFactory
from dl_cm.common.typing import StepOutputStruct, namedEntitySchema
from dl_cm.common.tasks.criterion import BaseLoss
from dl_cm.common.tasks.metrics import BaseMetric
from dl_cm.common.tasks.base_task import BaseTask
from dl_cm.common.learning.criterion_learner import CriterionLearner

class metricTrackCallback(pl.Callback, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            metrics: List[namedEntitySchema]
        return ValidConfig
    
    binary_train_metrics : MetricCollection
    binary_valid_metrics : MetricCollection
    loss_train_metrics : MetricCollection
    loss_valid_metrics : MetricCollection
    task_learner_criterion : BaseLoss
    
    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        pl.Callback.__init__(self)
        loaded_metrics = MetricsFactory.create(config.get("metrics"))
        self.extend_binary_metrics(loaded_metrics)
    
    def on_fit_start(self, trainer, pl_module: BaseTask):
        if isinstance(pl_module.learner, CriterionLearner):
            self.task_learner_criterion = pl_module.learner.criterion
            self.extend_loss_metrics(pl_module.learner.criterion)

    def extend_binary_metrics(self, metrics: BaseMetric | List[BaseMetric]):
        """
        Initializes the metrics collections for training and validation.
        This method sets up the general metrics for both training and validation phases.
        """
        metrics = metrics if isinstance(metrics, list) else [metrics]
        # General metrics (train)
        train_metrics = MetricCollection(metrics, prefix="train_")
        if self.binary_train_metrics is None:
            self.binary_train_metrics = train_metrics
        else:
            self.binary_train_metrics.add_metrics(train_metrics)
        
        # General metrics (valid)
        valid_metrics = train_metrics.clone(prefix="valid_")
        if self.binary_valid_metrics is None:
            self.binary_valid_metrics = valid_metrics
        else:
            self.binary_valid_metrics.add_metrics(valid_metrics)
    
    def extend_loss_metrics(self, criterion:BaseLoss):
        # Loss metrics initialization (train)
        loss_train_metrics : MetricCollection = \
              criterion.as_metric_collection().clone(prefix="train_")
        if self.loss_train_metrics is None:
            self.loss_train_metrics = loss_train_metrics
        else:
            self.loss_train_metrics.add_metrics(loss_train_metrics)
        # Loss metrics initialization (valid)
        loss_valid_metrics : MetricCollection = \
            self.loss_train_metrics.clone(prefix="valid_")
        if self.loss_valid_metrics is None:
            self.loss_valid_metrics = loss_valid_metrics
        else:
            self.loss_valid_metrics.add_metrics(loss_valid_metrics)
    
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: StepOutputStruct, batch, batch_idx: int
    ) -> None:
        self.binary_train_metrics.update(preds=outputs.predictions, target=outputs.targets)
        if isinstance(outputs.loss, dict):
            for (loss_name, loss_value) in outputs.loss.items():
                self.loss_train_metrics[loss_name].update(loss_value)
        else:
            self.loss_train_metrics[self.task_learner_criterion.name()].update(outputs.loss)

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: StepOutputStruct,
            batch,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.binary_valid_metrics.update(preds=outputs.predictions, target=outputs.targets)
        if isinstance(outputs.loss, dict):
            for (loss_name, loss_value) in outputs.loss.items():
                self.loss_valid_metrics[loss_name].update(loss_value)
        else:
            self.loss_valid_metrics[self.task_learner_criterion.name()].update(outputs.loss)
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule)-> None:
        self.binary_train_metrics.compute()
        self.loss_train_metrics.compute()
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule)-> None:
        self.binary_valid_metrics.compute()
        self.loss_valid_metrics.compute()
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule)-> None:
        self.binary_train_metrics.reset()
        self.loss_train_metrics.reset()
        
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule)-> None:
        self.binary_valid_metrics.reset()
        self.loss_valid_metrics.reset()