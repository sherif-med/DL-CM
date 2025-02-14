from dl_cm.utils.registery import Registry
from dl_cm.utils.ppattern.factory import BaseFactory
import copy
from dl_cm.common import DLCM
from typing import Type
from dl_cm.utils.ppattern.data_validation import validationMixin
from dl_cm.utils.ppattern.init_check_mixin import InitCheckMixin
from dl_cm.common.typing import namedEntitySchema
import pydantic as pd

DATASETS_REGISTERY = Registry("Datasets")
LOADED_DATASETS_REGISTRY = Registry("Loaded datasets")

class BaseDataset(DLCM, validationMixin, InitCheckMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(namedEntitySchema):
            reference_name: str = None # this field is used to differentiate between loaded datasets
        return ValidConfig

    @staticmethod
    def registry() -> Registry:
        return DATASETS_REGISTERY

    def __init__(self, config: dict) -> None:
        validationMixin.__init__(self, config)
        InitCheckMixin.__init__(self)
        ref_name = config.get("reference_name", self.__class__.__name__)
        if ref_name in LOADED_DATASETS_REGISTRY:
            raise ValueError(f"Dataset with name {ref_name} is already loaded")
        LOADED_DATASETS_REGISTRY.register(obj=self, name=ref_name)
        self._ref_name = ref_name
    
    @property
    def reference_name(self) -> str:
        self.check_base_class_initialized()
        return self._ref_name
    
    def compose(self, composition_cls: type["CompositionDataset"], **kwargs) -> "CompositionDataset":
        return composition_cls({"parent_dataset": self, "copy_parent": False, **kwargs})
    
class DatasetFactory(BaseFactory[BaseDataset]):

    @staticmethod
    def base_class()-> Type["BaseDataset"]:
        return BaseDataset

class CompositionDataset(BaseDataset, validationMixin, InitCheckMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            parent_dataset: namedEntitySchema | BaseDataset
            copy_parent: bool = False
        return ValidConfig

    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        InitCheckMixin.__init__(self)
        super().__init__(config)
        parent_dataset = DatasetFactory.create(parent_dataset)
        self._parent_dataset = copy.copy(parent_dataset) if config.get("copy_parent", False) else parent_dataset
    
    @property
    def parent_dataset(self) -> BaseDataset:
        self.check_base_class_initialized()
        return self._parent_dataset

    def parent_index(self, index):
        return index
    
    def top_parent_index(self, index):
        if not isinstance(self.parent_dataset, CompositionDataset):
            return self.parent_index(index)
        else:
            return self.parent_dataset.top_parent_index(self.parent_index(index)) 
    
    def get_top_dataset(self):
        if not isinstance(self.parent_dataset, CompositionDataset):
            return self.parent_dataset
        else:
            return self.parent_dataset.get_top_dataset()

from . import combined_dataset
from . import shuffled_dataset