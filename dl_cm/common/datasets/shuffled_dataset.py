from typing import List, Any, Callable, Union
from dl_cm.common.datasets import CompositionDataset
from dl_cm.common.datasets import BaseDataset
import numpy as np
from dl_cm.common.datasets import DATASETS_REGISTERY
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
from dl_cm.common.functions import FunctionsFactory

class ShuffledDataset(CompositionDataset, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            shuffled_indices: list[int] | np.ndarray | Callable | str = None
        return ValidConfig
    
    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        super().__init__(config)
        if isinstance(config.get("shuffled_indices"), (list, np.ndarray)):
            shuffled_indices = np.array(config.get("shuffled_indices"))
        elif callable(config.get("shuffled_indices")):
            shuffled_indices = config.get("shuffled_indices")(len(self.parent_dataset))
        elif isinstance(config.get("shuffled_indices"), str):
            shuffled_indices = FunctionsFactory.create(config.get("shuffled_indices"))
        else:
            shuffled_indices = np.random.permutation(len(self.parent_dataset))
        if len(shuffled_indices) != len(self.parent_dataset):
            raise Exception(f"Shuffled indices length {len(shuffled_indices)} is not equal to parent dataset length {len(self.parent_dataset)}")
        self.shuffled_indices = shuffled_indices
    
    def parent_index(self, index):
        return self.shuffled_indices[index]
        
    def __len__(self):
        return len(self.parent_dataset)
    
    def __getitem__(self, index):
        return self.parent_dataset[self.shuffled_indices[index]]