from typing import List

import pytorch_lightning as pl
from torchmetrics import MetricCollection

from dl_cm.common.learning.criterion_learner import CriterionLearner
from dl_cm.common.tasks import BaseTask
from dl_cm.common.tasks.criterion import BaseLoss
from dl_cm.common.tasks.metrics import BaseMetric, MetricsFactory
from dl_cm.common.trainer.callbacks import baseCallback
from dl_cm.common.typing import StepOutputStruct, namedEntitySchema


class metricTrackCallback(baseCallback):
    def __init__(self, metrics: list[namedEntitySchema | BaseMetric], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_train_metrics: MetricCollection = None
        self.binary_valid_metrics: MetricCollection = None
        self.loss_train_metrics: MetricCollection = None
        self.loss_valid_metrics: MetricCollection = None
        self.task_learner_criterion: BaseLoss = None
        loaded_metrics = MetricsFactory.create(metrics)
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
        metrics: List[BaseMetric] = metrics if isinstance(metrics, list) else [metrics]
        # General metrics (train)
        train_metrics = MetricCollection(
            {m.instance_name__(): m for m in metrics}, prefix="train_"
        )
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

    def extend_loss_metrics(self, criterion: BaseLoss):
        # Loss metrics initialization (train)
        loss_train_metrics: MetricCollection = criterion.as_metric_collection().clone(
            prefix="train_"
        )
        if self.loss_train_metrics is None:
            self.loss_train_metrics = loss_train_metrics
        else:
            self.loss_train_metrics.add_metrics(loss_train_metrics)
        # Loss metrics initialization (valid)
        loss_valid_metrics: MetricCollection = self.loss_train_metrics.clone(
            prefix="valid_"
        )
        if self.loss_valid_metrics is None:
            self.loss_valid_metrics = loss_valid_metrics
        else:
            self.loss_valid_metrics.add_metrics(loss_valid_metrics)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: StepOutputStruct,
        batch,
        batch_idx: int,
    ) -> None:
        self.binary_train_metrics.update(
            preds=outputs.predictions, target=outputs.targets
        )
        if isinstance(outputs.loss, dict):
            for loss_name, loss_value in outputs.loss.items():
                self.loss_train_metrics[loss_name].update(loss_value)
        else:
            self.loss_train_metrics[self.task_learner_criterion.name()].update(
                outputs.loss
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: StepOutputStruct,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.binary_valid_metrics.update(
            preds=outputs["predictions"], target=batch["targets"]
        )
        if isinstance(outputs.loss, dict):
            for loss_name, loss_value in outputs.loss.items():
                self.loss_valid_metrics[loss_name].update(loss_value)
        else:
            self.loss_valid_metrics[self.task_learner_criterion.name()].update(
                outputs.loss
            )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.binary_train_metrics.compute()
        self.loss_train_metrics.compute()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.binary_valid_metrics.compute()
        self.loss_valid_metrics.compute()

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.binary_train_metrics.reset()
        self.loss_train_metrics.reset()

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.binary_valid_metrics.reset()
        self.loss_valid_metrics.reset()
