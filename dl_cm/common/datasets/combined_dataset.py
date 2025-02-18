import bisect
import itertools

from dl_cm.common.datasets import BaseDataset, DatasetFactory
from dl_cm.common.typing import namedEntitySchema


class CombinedDataset(BaseDataset):

    def __init__(self, datasets: list[namedEntitySchema | BaseDataset], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasets : list[BaseDataset] = DatasetFactory.create(datasets)

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
