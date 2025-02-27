import torch

from dl_cm.common.datasets import CompositionDataset, IdentityCompositionDataset
from dl_cm.common.datasets.combined_dataset import CombinedDataset

from . import BaseSampler


class HeteroDatasetsBatchSampler(BaseSampler):
    """
    HeteroDatasetsBatchSampler: Sample batches of different sizes from datasets within CombinedDataset.

    The purpose of this sampler is to sample batches of different sizes from datasets within CombinedDataset.
    It will sample batches of size batch_sizes[i] from the i-th dataset in the CombinedDataset. The order of the datasets
    is determined by the order of the datasets in the CombinedDataset.

    The sampler will sample all the elements of the CombinedDataset and will not drop any element.

    Note: This sampler is the default sampler for CombinedDataset and CompositionDataset with top dataset as CombinedDataset.
    """

    def __init__(
        self,
        data_source,
        batch_sizes: tuple,
        *args,
        drop_last=False,
        shuffle=True,
        **kwargs,
    ):
        assert isinstance(data_source, CombinedDataset) or (
            isinstance(data_source, CompositionDataset)
            and isinstance(data_source.top_dataset, CombinedDataset)
        ), (
            "data_source argument should be either a CombinedDataset or a CompositionDataset with top dataset as CombinedDataset!"
        )
        if not isinstance(data_source, CompositionDataset):
            self.data_source = IdentityCompositionDataset[..., CombinedDataset](
                data_source, copy_parent=False
            )
        else:
            self.data_source = data_source
        self.batch_sizes = batch_sizes
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.combined_dataset_count = self._count_top_datasets_length()
        super().__init__(*args, **kwargs)

    def _count_top_datasets_length(self):
        counter_map = {k: 0 for k in range(len(self.data_source.top_dataset.datasets))}
        for c_idx in range(len(self.data_source)):
            idx_within_combined_dst = self.data_source.top_parent_index(c_idx)
            dataset_idx = self.data_source.top_dataset.respective_dataset_index(
                idx_within_combined_dst
            )
            counter_map[dataset_idx] += 1
        return counter_map

    def __iter__(self):
        batch_buffers: dict[
            int, list
        ] = {}  # To hold batches of different types (keys are the index of the respective dataset in the CombinedDataset)
        item_loop_indices = (
            torch.randperm(len(self.data_source)).tolist()
            if self.shuffle
            else range(len(self.data_source))
        )
        for idx in item_loop_indices:
            idx_within_combined_dst = self.data_source.top_parent_index(idx)
            dataset_idx = self.data_source.top_dataset.respective_dataset_index(
                idx_within_combined_dst
            )

            if dataset_idx not in batch_buffers:
                batch_buffers[dataset_idx] = []

            batch_buffers[dataset_idx].append(idx)  # Store indices instead of items
            # Use the specific batch size for the item type
            if len(batch_buffers[dataset_idx]) == self.batch_sizes[dataset_idx]:
                yield batch_buffers[dataset_idx]
                batch_buffers[dataset_idx] = []

        # Handling remaining items if drop_last is False
        if not self.drop_last:
            for dataset_idx, buffer in batch_buffers.items():
                if buffer:
                    yield buffer

    def __len__(self):
        total_length = sum(
            [
                self.combined_dataset_count[dst_idx] // bs
                for dst_idx, bs in enumerate(self.batch_sizes)
            ]
        )
        return total_length
