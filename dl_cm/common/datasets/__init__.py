from dl_cm.utils.registery import Registry
from dl_cm.config_loaders import load_named_entity
from dl_cm.utils.exceptions import OutOfTypesException
import copy

DATASETS_REGISTERY = Registry("Datasets")

class BaseDataset:
    pass

class DatasetFactory:
    def __init__(self):
        pass

    @classmethod
    def create(cls, c_dataset):
        if isinstance(c_dataset, str):
            dataset_class = DATASETS_REGISTERY.get(c_dataset)
            return dataset_class()
        elif isinstance(c_dataset, dict):
            return load_named_entity(DATASETS_REGISTERY, c_dataset)
        elif isinstance(c_dataset, BaseDataset):
            return c_dataset
        else:
            raise OutOfTypesException(c_dataset, (str, dict, BaseDataset,))


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