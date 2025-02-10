from dl_cm.common.tasks.base_task import StepOutputDict
from dl_cm.common.trainers.callbacks.metric_track_callback import metricTrackCallback
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
import pytorch_lightning as pl
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

class MetricsLoggingCallback(pl.Callback, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(LoggingOptions):
            defaults: defaulLoggingOptions
            train: list[namedLogModOptions] = None
            valid: list[namedLogModOptions] = None
        return ValidConfig

    def __init__(self, callback_config:dict):
        validationMixin.__init__(self, callback_config)
        pl.Callback.__init__(self)
        self.init_logging_flags(callback_config)
        self.metrics_dict = {
            "train": {},
            "valid": {}
        }
    
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
    
    
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        all_callbacks = trainer.callbacks
        callback_found = False
        for callback in all_callbacks:
            if isinstance(callback, metricTrackCallback):
                callback_found = True
                self.metrics_dict["train"] = callback.train_metrics
                self.metrics_dict["valid"] = callback.valid_metrics

        if not callback_found:
            logger.warning("No metricTrackCallback found in trainer callbacks")

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module, outputs: StepOutputDict, batch, batch_idx: int
    ) -> None:
        for (metric_name, metric) in self.metrics_dict["train"].items():
            self.log(f"train_{metric_name}", metric, 
                    on_step=self.logging_flags.get("train").get(metric_name).get("log_on_step"),
                    on_epoch=self.logging_flags.get("train").get(metric_name).get("log_on_epoch"))
        return

    def on_validation_batch_end(
            self, trainer: pl.Trainer, pl_module, outputs: StepOutputDict, batch, batch_idx: int
    ) -> None:
        for (metric_name, metric) in self.metrics_dict["valid"].items():
            self.log(f"valid_{metric_name}", metric, 
                         on_step=self.logging_flags.get("valid").get(metric_name).get("log_on_step"),
                         on_epoch=self.logging_flags.get("valid").get(metric_name).get("log_on_epoch"))
        return