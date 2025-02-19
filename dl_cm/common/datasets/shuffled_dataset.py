import collections
import collections.abc

import numpy as np

from dl_cm.common.datasets import COMPOSED_DATASET_CLASS, CompositionDataset


class ShuffledDataset(CompositionDataset[COMPOSED_DATASET_CLASS]):
    def __init__(
        self,
        shuffled_indices: collections.abc.Iterable[int] | np.ndarray = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(shuffled_indices, (collections.abc.Iterable[int], np.ndarray)):
            shuffled_indices = np.array(shuffled_indices)
        else:
            shuffled_indices = np.random.permutation(len(self.parent_dataset))
        if len(shuffled_indices) != len(self.parent_dataset):
            raise RuntimeError(
                f"Shuffled indices length {len(shuffled_indices)} \
                is not equal to parent dataset length {len(self.parent_dataset)}"
            )
        self.shuffled_indices: np.ndarray = shuffled_indices

    def parent_index(self, index):
        return self.shuffled_indices[index]

    def __len__(self):
        return len(self.parent_dataset)

    def __getitem__(self, index):
        return self.parent_dataset[self.shuffled_indices[index]]
