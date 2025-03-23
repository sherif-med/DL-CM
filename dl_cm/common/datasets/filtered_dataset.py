from typing import Callable

from dl_cm.common.datasets import (
    COMPOSED_DATASET_CLASS,
    TOP_DATASET_CLASS,
    CompositionDataset,
)
from dl_cm.common.functions import FunctionsFactory


class FilteredItemsDataset(
    CompositionDataset[COMPOSED_DATASET_CLASS, TOP_DATASET_CLASS]
):
    def __init__(self, filter_fn: str | Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.is_in_memory:
            raise TypeError(f"Expected ItemsDataset, got {type(self.parent_dataset)}")
        filter_fn = FunctionsFactory.create(filter_fn)
        self.filtered_items_indices = []

        for idx, item in enumerate(self.parent_dataset):
            if filter_fn(item):
                self.filtered_items_indices.append(idx)

    def __len__(self):
        return len(self.filtered_items_indices)

    def __getitem__(self, index):
        return self.parent_dataset[self.parent_index(index)]

    def parent_index(self, index):
        return self.filtered_items_indices[index]
