from typing import Callable

from dl_cm.common.data.datasets import (
    COMPOSED_DATASET_CLASS,
    TOP_DATASET_CLASS,
    CompositionDataset,
)
from dl_cm.common.data.transformations.general_transformation import (
    GeneralTransformationFactory,
)


class PreprocessedDataset(
    CompositionDataset[COMPOSED_DATASET_CLASS, TOP_DATASET_CLASS]
):
    def __init__(self, preprocessing_fn: str | Callable = "id", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessing_callable = GeneralTransformationFactory.create(
            preprocessing_fn
        )

    def __len__(self):
        return len(self.parent_dataset)

    def __getitem__(self, index):
        return self.preprocessing_callable(self.parent_dataset[index])

    def parent_index(self, index):
        return index
