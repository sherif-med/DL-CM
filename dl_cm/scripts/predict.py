"""
Prediction Script

This module provides a script for inference using the
Deep Learning Configuration Manager (DL-CM) framework.

It includes a [predict] click command that takes an input & output & strategy config, a checkpoint and
optionally accelerator option, and runs a model using the specified configuration.

The prediction process involves:

* Loading the configuration file and validating it against dlcm schema
* Creating a data module and a trainer instance based on the configuration
* Running the task using the trainer instance

Usage
-----

To use this script, run it from the command line and pass in the path to a
configuration file.

Example
-------

>>> python predict.py --config path/to/config.yaml --ckpt_path path/to/task.ckpt

"""

# pylint: disable=no-value-for-parameter
from pytorch_lightning import seed_everything

from dl_cm.common.data.datamodule import DataModulesFactory
from dl_cm.common.tasks import BaseTask
from dl_cm.common.trainer import load_trainer
from dl_cm.config_loaders import open_config_file
from dl_cm.scripts import BasePredictionCommand
from dl_cm.utils.config_validation import validate_config
from dl_cm import PREDICTION_SCHEMA_PATH

@BasePredictionCommand
def predict(config_path: str, ckpt_path: str):
    """
    Run prediction on a dataset using model saved within the checkpoint.

    This command loads a configuration file and uses it to create a data module
    and a trainer instance. It then runs inference the task using the trainer instance.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    ckpt_path : str
        Path to the checkpoint to be loaded.

    """
    validate_config(config_path, PREDICTION_SCHEMA_PATH)
    config: dict = open_config_file(config_path)
    loaded_datamodule = DataModulesFactory.create(config.get("datamodule"))
    loaded_trainer = load_trainer(**config.get("trainer"))
    loaded_task = BaseTask.load_from_checkpoint(ckpt_path)
    loaded_task.to(loaded_trainer.strategy.root_device)
    loaded_trainer.predict(loaded_task, datamodule=loaded_datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    predict()
