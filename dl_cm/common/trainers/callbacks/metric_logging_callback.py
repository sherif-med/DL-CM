from dl_cm.common.typing import StepOutputStruct
from dl_cm.common.trainers.callbacks.metric_track_callback import metricTrackCallback
from dl_cm.common.tasks.base_task import BaseTask
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
import pytorch_lightning as pl
from dl_cm.common.trainers.callbacks import BaseCallback
from dl_cm import _logger as logger

class logModOptions(pd.BaseModel):
    log_on_step: bool = None
    log_on_epoch: bool = None

class namedLogModOptions(logModOptions):
    name: str

class defaulLoggingOptions(pd.BaseModel):
    train: logModOptions
    valid: logModOptions

class LoggingOptions(pd.BaseModel):
    train: list[namedLogModOptions]
    valid: list[namedLogModOptions]

class MetricsLoggingCallback(BaseCallback, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(LoggingOptions):
            defaults: defaulLoggingOptions
            train: list[namedLogModOptions] = None
            valid: list[namedLogModOptions] = None
        return ValidConfig

    metric_track_callback : metricTrackCallback

    def __init__(self, callback_config:dict):
        validationMixin.__init__(self, callback_config)
        super().__init__(self)
        self.init_logging_flags(callback_config)
    
    def init_logging_flags(self, logging_config:dict):
        self.default_logging_flags = defaulLoggingOptions(**logging_config.get("defaults"))

        self.logging_flags : LoggingOptions = {
            "train": {
                **{c_log_mode.get("name"): self.default_logging_flags.get("train") | c_log_mode for c_log_mode in logging_config.get("train", [])}
            },
            "valid": {
                **{c_log_mode.get("name"): self.default_logging_flags.get("valid") | c_log_mode for c_log_mode in logging_config.get("val", [])}
            }
        }
    
    def log(self, pl_module: pl.LightningModule, metric_name: str, metric_value: float, mode: str):
        pl_module.log(metric_name, metric_value, 
                      on_step=self.logging_flags.get(mode).get(metric_name).get("log_on_step"),
                        on_epoch=self.logging_flags.get(mode).get(metric_name).get("log_on_epoch")
                        )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        all_callbacks = trainer.callbacks
        for callback in all_callbacks:
            if isinstance(callback, metricTrackCallback):
                if not self.metric_track_callback:
                    self.metric_track_callback = callback
                else:
                    logger.warning("Multiple metricTrackCallbacks found in trainer callbacks")

        if not self.metric_track_callback:
            logger.critical("No metricTrackCallback found in trainer callbacks")

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: BaseTask, outputs: StepOutputStruct, batch, batch_idx: int
    ) -> None:
        if not self.metric_track_callback:
            logger.warning("No metricTrackCallback found in trainer callbacks! Not logging metrics")
            return
        # General metrics logging
        for (metric_name, metric) in self.metric_track_callback.binary_train_metrics.items():
            self.log(pl_module, f"train_{metric_name}", metric, "train")
        # Loss metrics logging
        for (metric_name, metric) in self.metric_track_callback.loss_train_metrics.items():
            self.log(pl_module, f"train_{metric_name}", metric, "train")

    def on_validation_batch_end(
            self, trainer: pl.Trainer, pl_module : pl.LightningModule, outputs: StepOutputStruct, batch, batch_idx: int
    ) -> None:
        if not self.metric_track_callback:
            logger.warning("No metricTrackCallback found in trainer callbacks! Not logging metrics")
            return
        # General metrics logging
        for (metric_name, metric) in self.metric_track_callback.binary_valid_metrics.items():
            self.log(pl_module, f"valid_{metric_name}", metric, "valid")
        # Loss metrics logging
        for (metric_name, metric) in self.metric_track_callback.loss_valid_metrics.items():
            self.log(pl_module, f"valid_{metric_name}", metric, "valid")