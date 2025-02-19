import collections
import collections.abc

import numpy as np

from dl_cm import _logger as logger
from dl_cm.common.datasets import (
    COMPOSED_DATASET_CLASS,
    BaseDataset,
    CompositionDataset,
)
from dl_cm.common.typing import namedEntitySchema


class SubDataset(CompositionDataset[COMPOSED_DATASET_CLASS]):
    def __init__(
        self,
        parent_dataset: BaseDataset | namedEntitySchema,
        indices: collections.abc.Iterable[int] | np.ndarray[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(parent_dataset, *args, **kwargs)
        if indices is None:
            logger.warning("No indices provided for subdataset, using all indices!")
            self.indices = list(range(len(self.parent_dataset)))
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def parent_index(self, index: int):
        return self.indices[index]

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        return self.parent_dataset.__getitem__(real_idx)
