from dl_cm.utils.registery import Registry
from dl_cm.utils.ppattern.factory import BaseFactory
import copy

DATASETS_REGISTERY = Registry("Datasets")

class BaseDataset:
    pass

class DatasetFactory(BaseFactory):

    @classmethod
    def base_class(cls)-> type:
        return BaseDataset
    
    @classmethod
    def registry(cls) -> Registry:
        return DATASETS_REGISTERY


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