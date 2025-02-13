from typing import Callable
from dl_cm.common.datasets import CompositionDataset
from dl_cm.common.datasets.filtered_dataset import ItemsDataset
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
from dl_cm.common.functions import FunctionsFactory

class OrderedItemsDataset(CompositionDataset, ItemsDataset, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            order_fn: str | Callable
        return ValidConfig
    
    def __init__(self, config:dict):
        validationMixin.__init__(self, config)
        super().__init__(config)
        element_to_value = FunctionsFactory.create(config.get("order_fn"))
        indices = list(range(len(self.parent_dataset)))
        self.reordered_items_indices = sorted(indices, key=lambda index: element_to_value(self.parent_dataset[index]))
        
    def __len__(self):
        return len(self.parent_dataset)
    
    def __getitem__(self, index):
        return self.parent_dataset[self.reordered_items_indices[index]]