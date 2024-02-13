from dl_cm.utils.registery import Registry
import copy

class CompositionDataset(object):

    def __init__(self, parent_dataset, copy_parent=True):
        self.parent_dataset = copy.copy(parent_dataset) if copy_parent else parent_dataset
    
    def get_top_dataset(self):
        if not isinstance(self.parent_dataset, CompositionDataset):
            return self.parent_dataset
        else:
            return self.parent_dataset.get_top_dataset()


DATASETS_REGISTERY = Registry("Datasets")