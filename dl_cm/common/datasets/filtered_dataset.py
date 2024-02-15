from dl_cm.common.datasets import CompositionDataset
from dl_cm.common.datasets.items_dataset import ItemsDataset
from typing import Callable

class FilteredItemsDataset(CompositionDataset):
    
    def __init__(self, parent_dataset:ItemsDataset, filter_fn:Callable):
        assert isinstance(parent_dataset, ItemsDataset), "parent_dataset type expected to be ItemsDataset"
        super().__init__(parent_dataset, copy_parent=False)
        self.filtered_items_indices = []
        
        for idx, item in enumerate(self.parent_dataset.items):
            if filter_fn(item):
                self.filtered_items_indices.append(idx)
        
    def __len__(self):
        return len(self.filtered_items_indices)
    
    def __getitem__(self, index):
        return self.parent_dataset[self.filtered_items_indices[index]]
