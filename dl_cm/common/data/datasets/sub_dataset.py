import collections
import collections.abc

import numpy as np

from dl_cm import _logger as logger
from dl_cm.common.data.datasets import (
    COMPOSED_DATASET_CLASS,
    TOP_DATASET_CLASS,
    BaseDataset,
    CompositionDataset,
)
from dl_cm.common.typing import namedEntitySchema


class SubDataset(CompositionDataset[COMPOSED_DATASET_CLASS, TOP_DATASET_CLASS]):
    def __init__(
        self,
        parent_dataset: BaseDataset | namedEntitySchema,
        indices: collections.abc.Iterable[int] | np.ndarray[int] = None,
        bounds: tuple = None,
        *args,
        **kwargs,
    ):
        super().__init__(parent_dataset, *args, **kwargs)
        if indices is None:
            if bounds:
                assert len(bounds)==2, "Bounds size is different than 2"
                start_bound = self.validate_bound(bounds[0])
                end_bound = self.validate_bound(bounds[1])
                assert end_bound > start_bound, "End bound should be higher than start bound!"
                self.indices = list(range(start_bound, end_bound))
            else:
                logger.warning("No indices provided for subdataset, using all indices!")
                self.indices = list(range(len(self.parent_dataset)))
        else:
            assert bounds is None, "bounds and indices arguments cannot be both set!"
            self.indices = indices

    def validate_bound(self, bound)->int:
        if bound < 1 and bound > -1:
            bound = int(len(self.parent_dataset) * bound)
        if bound < 0:
            bound = bound + len(self.parent_dataset)
        if bound > len(self.parent_dataset):
            raise ValueError("Bound should be less than dataset length!")
        return bound


    def __len__(self):
        return len(self.indices)

    def parent_index(self, index: int):
        return self.indices[index]

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        return self.parent_dataset.__getitem__(real_idx)
