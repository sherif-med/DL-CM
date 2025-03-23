from typing import Callable

from dl_cm.common.datasets import (
    COMPOSED_DATASET_CLASS,
    TOP_DATASET_CLASS,
    CompositionDataset,
)
from dl_cm.common.functions import FunctionsFactory


class OrderedItemsDataset(
    CompositionDataset[COMPOSED_DATASET_CLASS, TOP_DATASET_CLASS]
):
    def __init__(self, order_fn: str | Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        element_to_value = FunctionsFactory.create(order_fn)
        indices = list(range(len(self.parent_dataset)))
        self.reordered_items_indices = sorted(
            indices, key=lambda index: element_to_value(self.parent_dataset[index])
        )

    def __len__(self):
        return len(self.parent_dataset)

    def __getitem__(self, index):
        return self.parent_dataset[self.reordered_items_indices[index]]
