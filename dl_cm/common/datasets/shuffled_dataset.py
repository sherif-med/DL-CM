from typing import List, Any, Callable, Union
from dl_cm.common.datasets import CompositionDataset
from dl_cm.common.datasets import BaseDataset
import numpy as np
from dl_cm.common.datasets import DATASETS_REGISTERY


class ShuffledDataset(CompositionDataset):
    
    def __init__(self, parent_dataset:BaseDataset, shuffled_indices:Union[list, np.ndarray] = None):
        super().__init__(parent_dataset, copy_parent=False)
        if shuffled_indices:
            assert len(self.parent_dataset) == len(shuffled_indices)
            self.shuffled_indices = shuffled_indices
        else:
            self.shuffled_indices = np.random.permutation(len(self.parent_dataset))
    
    def parent_index(self, index):
        return self.shuffled_indices[index]
        
    def __len__(self):
        return len(self.parent_dataset)
    
    def __getitem__(self, index):
        return self.parent_dataset[self.shuffled_indices[index]]