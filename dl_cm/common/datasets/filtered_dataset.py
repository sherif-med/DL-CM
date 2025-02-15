from dl_cm.common.datasets import CompositionDataset
from dl_cm.common.datasets.items_dataset import ItemsDataset
import pydantic as pd
from dl_cm.utils.ppattern.data_validation import validationMixin
from dl_cm.common.functions import FunctionsFactory
from typing import Callable

class FilteredItemsDataset(CompositionDataset, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            filter_fn: str | Callable
        return ValidConfig

    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        super().__init__(config)
        if not isinstance(self.parent_dataset, ItemsDataset):
            raise TypeError(f"Expected ItemsDataset, got {type(self.parent_dataset)}")
        filter_fn = FunctionsFactory.create(config.get("filter_fn"))
        self.filtered_items_indices = []

        for idx, item in enumerate(self.parent_dataset):
            if filter_fn(item):
                self.filtered_items_indices.append(idx)

    def __len__(self):
        return len(self.filtered_items_indices)

    def __getitem__(self, index):
        return self.parent_dataset[self.filtered_items_indices[index]]
