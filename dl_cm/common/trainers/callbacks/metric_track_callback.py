import pytorch_lightning as pl
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
from typing import List
from torchmetrics import MetricCollection
from dl_cm.common.tasks.metrics import MetricsFactory
from dl_cm.common.tasks.base_task import StepOutputDict

class metricTrackCallback(pl.Callback, validationMixin):

    def config_schema(cls)-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            metrics: List[dict]
            target_key: str
            predicted_key: str
        return ValidConfig
    
    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        pl.Callback.__init__(self)
        self.target_key = config.get("target_key")
        self.predicted_key = config.get("predicted_key")
        self.train_metrics : MetricCollection
        self.valid_metrics : MetricCollection
        self.init_metrics()

    def init_metrics(self):
        """
        Initializes the metrics collections for training and validation.
        This method sets up the general metrics for both training and validation phases.
        """        
        # General metrics (train)
        train_metrics = MetricCollection(MetricsFactory.create(self.task_config.get("metrics")), prefix="train_")
        if self.train_metrics is None:
            self.train_metrics = train_metrics
        else:
            self.train_metrics.add_metrics(train_metrics)
        
        # General metrics (valid)
        valid_metrics = train_metrics.clone(prefix="valid_")
        if self.valid_metrics is None:
            self.valid_metrics = valid_metrics
        else:
            self.valid_metrics.add_metrics(valid_metrics)
    
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: StepOutputDict, batch, batch_idx: int
    ) -> None:
        preds = outputs["prediction"][self.predicted_key]
        targets = outputs["target"][self.target_key]
        self.train_metrics.update(preds=preds, target=targets)
        return

    def on_validation_batch_end(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: StepOutputDict, batch, batch_idx: int
    ) -> None:
        preds = outputs["prediction"][self.predicted_key]
        targets = outputs["target"][self.target_key]
        self.valid_metrics.update(preds=preds, target=targets)
        return