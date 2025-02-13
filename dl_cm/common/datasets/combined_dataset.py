from dl_cm.common.datasets import BaseDataset, DatasetFactory
import bisect
import itertools
import pydantic as pd
from dl_cm.utils.ppattern.data_validation import validationMixin
from dl_cm.common.typing import namedEntitySchema


class CombinedDataset(BaseDataset, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            datasets: list[namedEntitySchema]
        return ValidConfig
    
    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        super().__init__(config)
        self.datasets : list[BaseDataset] = DatasetFactory.create(config.get("datasets"))
        
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

