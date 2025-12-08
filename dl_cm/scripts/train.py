"""
Training Script

This module provides a script for training deep learning models using the
Deep Learning Configuration Manager (DL-CM) framework.

It includes a [train] click command that takes a configuration file and a checkpoint
path as input, and trains a model using the specified configuration.

The training process involves:

* Loading the configuration file and validating it against dlcm schema
* Creating a data module and a trainer instance based on the configuration
* Training the task using the trainer instance

Usage
-----

To use this script, run it from the command line and pass in the path to a
configuration file and a checkpoint path.

Example
-------

>>> python train.py --config path/to/config.yaml --ckpt path/to/checkpoint

"""

# pylint: disable=no-value-for-parameter
from lightning import seed_everything

from dl_cm.common.data.datamodule import DataModulesFactory
from dl_cm.common.tasks import TasksFactory
from dl_cm.common.trainer import load_trainer
from dl_cm.config_loaders import open_config_file
from dl_cm.scripts import BaseTrainingCommand
from dl_cm.utils.config_validation import validate_config


@BaseTrainingCommand
def train(config_path: str, ckpt_path: str, seed: int = -1):
    """
    Train a model using the specified configuration.

    This command loads a configuration file and uses it to create a data module
    and a trainer instance. It then trains the task using the trainer instance.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    ckpt_path : str
        Path to the checkpoint to be saved.
    seed : int, optional
        Random seed to use for training. If -1, no seeding is performed.

    """
    if seed != -1:
        seed_everything(seed, workers=True)
    validate_config(config_path)
    config: dict = open_config_file(config_path)
    loaded_datamodule = DataModulesFactory.create(config.get("datamodule"))
    loaded_trainer = load_trainer(**config.get("trainer"))
    loaded_task = TasksFactory.create(config.get("task"))
    loaded_task.to(loaded_trainer.strategy.root_device)
    loaded_trainer.fit(loaded_task, datamodule=loaded_datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    train()
