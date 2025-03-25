"""
Configuration Validation Module

This module provides utilities for validating configuration files used in the
Deep Learning Configuration Manager (DL-CM) framework.

It includes classes and functions for defining and validating configuration
schemas, as well as registering and loading configuration validators.

The main classes and functions in this module are:

* [RegistryValidator]: A custom validator for checking named entities against a registry.
* [validate_config]: A function for validating a configuration file against a schema.

Usage
-----

To use this module, import the [validate_config]: function and pass in the path to a
configuration file and a schema definition.

Example
-------

>>> from dl_cm.utils.config_validation import validate_config
>>> validate_config('path/to/config.yaml', 'path/to/schema.yaml')

"""

from functools import partial

import yamale
from pydantic import ConfigDict, validate_call
from yamale.validators import DefaultValidators, Validator

from dl_cm import DEFAULT_SCHEMA_PATH
from dl_cm import _logger as logger
from dl_cm.common.data.datamodule import DATAMODULES_REGISTERY
from dl_cm.common.data.datasets import DATASETS_REGISTERY
from dl_cm.common.data.transformations import TRANSFORMATION_REGISTRY
from dl_cm.common.learning import LEARNERS_REGISTERY
from dl_cm.common.learning.optimizable_learner import (
    LR_SCHEDULER_REGiSTERY,
    OPTIMIZER_REGiSTERY,
)
from dl_cm.common.models import MODELS_REGISTERY
from dl_cm.common.tasks import TASKS_REGISTERY
from dl_cm.common.tasks.criterion import CRITIREON_REGISTRY
from dl_cm.common.tasks.metrics import METRICS_REGISTRY
from dl_cm.common.trainer.callbacks import CALLBACKS_REGISTERY
from dl_cm.common.trainer.loggers import LOGGERS_REGISTERY
from dl_cm.config_loaders import open_config_file


class RegistryValidator(Validator):
    """Custom validator for checking named entities against a registry."""

    def __init__(self, name, registry, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.registry = registry

    @property
    def tag(self):
        return self.name

    def _is_valid(self, value):
        return value in self.registry


_registries_validators = {
    "registered_task": TASKS_REGISTERY,
    "registered_logger": LOGGERS_REGISTERY,
    "registered_dataset": DATASETS_REGISTERY,
    "registered_callback": CALLBACKS_REGISTERY,
    "registered_optimizer": OPTIMIZER_REGiSTERY,
    "registered_transformation": TRANSFORMATION_REGISTRY,
    "registered_augtrans": TRANSFORMATION_REGISTRY,
    "registered_critireon": CRITIREON_REGISTRY,
    "registered_metric": METRICS_REGISTRY,
    "registered_scheduler": LR_SCHEDULER_REGiSTERY,
    "registered_model": MODELS_REGISTERY,
    "registered_learner": LEARNERS_REGISTERY,
    "registered_datamodule": DATAMODULES_REGISTERY,
}

_extended_validators = DefaultValidators.copy()
for key, val_reg in _registries_validators.items():
    _extended_validators[key] = partial(RegistryValidator, key, val_reg)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def validate_config(
    config_path: str,
    schema_path: str = DEFAULT_SCHEMA_PATH,
    extra_validators: dict[str, Validator] = None,
):
    """
    Validate a configuration file against a given schema.

    :param config_path: path to a yaml configuration file
    :param schema_path: path to a yaml schema file (default to DEFAULT_SCHEMA_PATH)
    :param extra_validators: a dictionary of custom validators to add to the default ones
    """
    if extra_validators is None:
        extra_validators = _extended_validators
    config_schema = yamale.make_schema(
        schema_path,
        validators=extra_validators,
    )

    data = [(open_config_file(config_path), config_path)]

    try:
        yamale.validate(config_schema, data)
        logger.info("Validation success! 👍")
    except yamale.YamaleError as e:
        logger.info("Validation failed!\n")
        for result in e.results:
            logger.error(f"validating data '{result.data}' with '{result.schema}'")
            for error in result.errors:
                logger.error(f"{error}")
        exit(1)
