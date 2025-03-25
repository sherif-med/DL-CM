## Torch proposes a function to do this

import collections
import collections.abc

import numpy as np
import pydantic as pd

from dl_cm.common.data.datasets import BaseDataset
from dl_cm.common.data.datasets.items_dataset import ItemsDataset
from dl_cm.common.data.datasets.sub_dataset import SubDataset

from . import COMPOSED_DATASET_CLASS, TOP_DATASET_CLASS, CompositionDataset


class SplitDataset(CompositionDataset[COMPOSED_DATASET_CLASS, TOP_DATASET_CLASS]):
    @staticmethod
    def config_schema() -> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            reference_names: collections.abc.Iterable[str]
            split_ratios: list[float]

        return ValidConfig

    def __init__(
        self,
        reference_names: collections.abc.Iterable[str],
        split_ratios: collections.abc.Iterable[float],
        *args,
        **kwargs,
    ):
        self.reference_names = reference_names
        self.split_ratios = split_ratios

        if len(self.reference_names) != len(self.split_ratios):
            raise ValueError(
                "Parameters 'reference_names' and 'split_ratios'\
                should have the same length!"
            )
        super().__init__(*args, **kwargs)
        self._ref_datasets_map = {
            k: v
            for k, v in zip(
                self.reference_names,
                split_dataset_random(
                    self.parent_dataset, self.split_ratios, self.reference_names
                ),
            )
        }

    def get_dataset_by_ref_name(self, ref_name: str) -> BaseDataset:
        return self._ref_datasets_map.get(ref_name)


def split_dataset_random(
    parent_dataset: ItemsDataset,
    fractions: collections.abc.Iterable[float],
    reference_names: collections.abc.Iterable[str],
) -> list[SubDataset]:
    """
    Split a parent dataset into non-overlapping subdatasets with random indices,
    based on specified fractions.

    Parameters:
    - parent_dataset: The dataset to be split.
    - fractions: A list of fractions indicating the size of each subdataset relative to the parent dataset.

    Returns:
    - A list of subdatasets, each being a portion of the parent dataset with random indices.
    """
    assert sum(fractions) <= 1, "The sum of the fractions must not exceed 1."

    total_length = len(parent_dataset)
    indices = np.arange(0, total_length, 1)

    # Calculate the number of elements for each fraction
    counts = [int(frac * total_length) for frac in fractions]

    # Calculate the start index for each subdataset
    starts = np.cumsum([0] + counts)
    ends = starts[1:]

    subdatasets = []
    for start, end, ref_name in zip(starts, ends, reference_names):
        # Extract the indices for this subdataset
        sub_indices = indices[start:end]
        c_config = {
            "parent_dataset": parent_dataset,
            "indices": sub_indices,
            "reference_name": ref_name,
        }
        # Create a new subdataset instance for the current subset of indices
        subdatasets.append(SubDataset(c_config))

    return subdatasets
