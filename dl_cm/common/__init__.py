
from .datasets.transformations import TRANSFORMATION_REGISTRY
from .datasets import DATASETS_REGISTERY
from .datasets.preprocessed_dataset import PREPROCESSING_REGISTERY

from .trainers.callbacks import CALLBACKS_REGISTERY
from .trainers.loggers import LOGGERS_REGISTERY

from .tasks import TASKS_REGISTERY
from .tasks.optimizer import OPTIMIZER_REGiSTERY, LR_SCHEDULER_REGiSTERY
from .tasks.criterion import CRITIREON_REGISTRY
from .tasks.metrics import METRICS_REGISTRY

from .models import MODELS_REGISTERY

from dl_cm.utils.ppattern.self_registered_mixin import selfRegisteredMixin

class DLCM(selfRegisteredMixin):

    @staticmethod
    def base_class_adapter(external_cls: type, base_cls: type)-> type:
        """Decorator to make any external class inherit from BaseClass"""
        class WrappedClass(external_cls, base_cls):
            def __init__(self, config:dict):
                base_cls.__init__(self, config)
                external_cls.__init__(config.get("params", {}))
        return WrappedClass