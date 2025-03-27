from typing import Optional

import pydantic as pd
import pytorch_lightning as pl

from dl_cm import _logger as logger
from dl_cm.common.tasks import BaseTask
from dl_cm.common.trainer.callbacks import baseCallback
from dl_cm.common.trainer.callbacks.metric_track_callback import metricTrackCallback
from dl_cm.common.typing import StepOutputStruct


class logModOptions(pd.BaseModel):
    log_on_step: bool = False
    log_on_epoch: bool = True


class namedLogModOptions(logModOptions):
    name: str


class defaulLoggingOptions(pd.BaseModel):
    train: logModOptions = logModOptions()
    valid: logModOptions = logModOptions()


class LoggingOptions(pd.BaseModel):
    train: list[namedLogModOptions]
    valid: list[namedLogModOptions]


class MetricsLoggingCallback(baseCallback):
    def __init__(
        self,
        defaults=defaulLoggingOptions(),
        train: list[namedLogModOptions] = None,
        valid: list[namedLogModOptions] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.default_logging_flags = defaults
        self._metric_track_callback: Optional[metricTrackCallback] = None
        self.logging_flags: LoggingOptions = {
            "train": {
                **{
                    c_log_mode.get("name"): self.default_logging_flags.get("train")
                    | c_log_mode
                    for c_log_mode in train or []
                }
            },
            "valid": {
                **{
                    c_log_mode.get("name"): self.default_logging_flags.get("valid")
                    | c_log_mode
                    for c_log_mode in valid or []
                }
            },
        }

    def log(
        self,
        pl_module: pl.LightningModule,
        metric_name: str,
        metric_value: float,
        mode: str,
    ):
        pl_module.log(
            metric_name,
            metric_value,
            on_step=self.logging_flags.get(mode).get(metric_name).get("log_on_step"),
            on_epoch=self.logging_flags.get(mode).get(metric_name).get("log_on_epoch"),
        )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        all_callbacks = trainer.callbacks
        for callback in all_callbacks:
            if isinstance(callback, metricTrackCallback):
                if not self._metric_track_callback:
                    self._metric_track_callback = callback
                else:
                    logger.warning(
                        "Multiple metricTrackCallbacks found in trainer callbacks"
                    )

        if not self._metric_track_callback:
            logger.critical("No metricTrackCallback found in trainer callbacks")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseTask,
        outputs: StepOutputStruct,
        batch,
        batch_idx: int,
    ) -> None:
        if not self._metric_track_callback:
            logger.warning(
                "No metricTrackCallback found in trainer callbacks! Not logging metrics"
            )
            return
        # General metrics logging
        for (
            metric_name,
            metric,
        ) in self._metric_track_callback.binary_train_metrics.items():
            self.log(pl_module, f"train_{metric_name}", metric, "train")
        # Loss metrics logging
        for (
            metric_name,
            metric,
        ) in self._metric_track_callback.loss_train_metrics.items():
            self.log(pl_module, f"train_{metric_name}", metric, "train")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: StepOutputStruct,
        batch,
        batch_idx: int,
    ) -> None:
        if not self._metric_track_callback:
            logger.warning(
                "No metricTrackCallback found in trainer callbacks! Not logging metrics"
            )
            return
        # General metrics logging
        for (
            metric_name,
            metric,
        ) in self._metric_track_callback.binary_valid_metrics.items():
            self.log(pl_module, f"valid_{metric_name}", metric, "valid")
        # Loss metrics logging
        for (
            metric_name,
            metric,
        ) in self._metric_track_callback.loss_valid_metrics.items():
            self.log(pl_module, f"valid_{metric_name}", metric, "valid")
