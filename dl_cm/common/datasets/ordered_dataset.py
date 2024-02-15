from typing import List, Any, Callable
from dl_cm.common.datasets import CompositionDataset
from dl_cm.common.datasets.filtered_dataset import ItemsDataset

class OrderedItemsDataset(CompositionDataset):
    
    def __init__(self, parent_dataset:ItemsDataset, element_to_value:Callable):
        super().__init__(parent_dataset, copy_parent=False)        
        indices = list(range(len(self.parent_dataset)))
        self.reordered_items_indices = sorted(indices, key=lambda index: element_to_value(self.parent_dataset[index]))
        
        
    def __len__(self):
        return len(self.parent_dataset)
    
    def __getitem__(self, index):
        return self.parent_dataset[self.reordered_items_indices[index]]