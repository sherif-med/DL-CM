from dl_cm.common.datasets import BaseDataset, DatasetFactory, DATASETS_REGISTERY
import bisect
import itertools
from collections.abc import Iterable
from typing import Union


@DATASETS_REGISTERY.register()
class CombinedDataset(BaseDataset):
    
    def __init__(self, datasets_def: Iterable[Union[BaseDataset, dict, str]]):
        
        self.datasets = [ DatasetFactory.create(c_dst_def) for c_dst_def in datasets_def ]
        
        self.datasets_lengths = [len(d) for d in self.datasets]
        self.cumulative_lengths = [0] + list(itertools.accumulate(self.datasets_lengths))
        
        self.total_length = sum(self.datasets_lengths)
    
    
    def __len__(self):
        return self.total_length
    
    def respective_dataset_index(self, index):
        dataset_idx = bisect.bisect_right(self.cumulative_lengths, index) - 1
        return dataset_idx
           
    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)

        if idx >= self.cumulative_lengths[-1]:
            raise IndexError("Index out of range")

        # Use binary search to find the appropriate dataset
        dataset_idx = self.respective_dataset_index(idx)
        dataset_offset = self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][idx - dataset_offset]

