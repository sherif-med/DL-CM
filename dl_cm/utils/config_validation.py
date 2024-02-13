import yamale
from dl_cm import get_schema_path
from functools import partial
from yamale.validators import DefaultValidators, Validator

from dl_cm.common import TASKS_REGISTERY, LOGGERS_REGISTERY, DATASETS_REGISTERY, \
    CALLBACKS_REGISTERY, OPTIMIZER_REGiSTERY, PREPROCESSING_REGISTERY, AUGMENTATION_TRANSFORMATION_REGISTERY

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

registries_validators = {
    "registered_task": TASKS_REGISTERY,
    "registered_logger": LOGGERS_REGISTERY,
    "registered_dataset": DATASETS_REGISTERY,
    "registered_callback": CALLBACKS_REGISTERY,
    "registered_optimizer": OPTIMIZER_REGiSTERY,
    "registered_preprocessing": PREPROCESSING_REGISTERY,
    "registered_augtrans": AUGMENTATION_TRANSFORMATION_REGISTERY,
}


def extended_validators():
    """
    Returns a list of validators that combines default and regstries validators!    
    """
    validators = DefaultValidators.copy()
    for key, val_reg in registries_validators.items():
        validators[key] = partial(RegistryValidator, key, val_reg)
    return validators
    

def validate_config(config_path, registry_validation=False):
    """"""
    config_schema = yamale.make_schema(get_schema_path(), validators=extended_validators() if registry_validation else None)
    
    data = yamale.make_data(config_path)

    try:
        yamale.validate(config_schema, data)
        print('Validation success! 👍')
    except yamale.YamaleError as e:
        print('Validation failed!\n')
        for result in e.results:
            print("Error validating data '%s' with '%s'\n\t" % (result.data, result.schema))
            for error in result.errors:
                print('\t%s' % error)
        exit(1)