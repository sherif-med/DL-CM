from dl_cm.utils.registery import Registry
from dl_cm.utils.ppattern.factory import BaseFactory
import copy
from dl_cm.common import DLCM
from typing import Type
from dl_cm.utils.ppattern.data_validation import validationMixin
from dl_cm.common.typing import namedEntitySchema
import pydantic as pd

DATASETS_REGISTERY = Registry("Datasets")
class BaseDataset(DLCM, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        return namedEntitySchema

    @staticmethod
    def registry() -> Registry:
        return DATASETS_REGISTERY

    def __init__(self, config: dict) -> None:
        validationMixin.__init__(self, config)
    
class DatasetFactory(BaseFactory):

    @staticmethod
    def base_class()-> Type["DLCM"]:
        return BaseDataset


class CompositionDataset(BaseDataset):

    def __init__(self, parent_dataset, copy_parent=True):
        parent_dataset = DatasetFactory.create(parent_dataset)
        self.parent_dataset = copy.copy(parent_dataset) if copy_parent else parent_dataset
    
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