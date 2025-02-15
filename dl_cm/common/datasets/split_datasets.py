## Torch proposes a function to do this

import collections.abc
import numpy as np
from . import CompositionDataset
import collections
from dl_cm.common.datasets.items_dataset import ItemsDataset
from dl_cm.common.datasets import BaseDataset
import pydantic as pd
from dl_cm.utils.ppattern.data_validation import validationMixin
from dl_cm.common.typing import namedEntitySchema
from dl_cm import _logger as logger

# Define a subdataset class that uses these indices
class SubDataset(CompositionDataset, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(namedEntitySchema):
            indices: list[int] | np.ndarray = None
        return ValidConfig

    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        super().__init__(config.get("parent_dataset"))
        if config.get("indices") is None:
            logger.warning("No indices provided for subdataset, using all indices!")
            self.indices = list(range(len(self.parent_dataset)))
        else:
            self.indices = config.get("indices")

    def __len__(self):
        return len(self.indices)

    def parent_index(self, index: int):
        return self.indices[index]

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        return self.parent_dataset.__getitem__(real_idx)

class SplitDataset(CompositionDataset, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            reference_names: list[str]
            split_ratios: list[float]
        return ValidConfig

    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        self.reference_names = config.get("reference_names")
        self.split_ratios = config.get("split_ratios")

        if (len(self.reference_names)!= len(self.split_ratios)):
            raise ValueError("Parameters 'reference_names' and 'split_ratios'\
                should have the same length!")
        super().__init__(config.get("parent_dataset"))
        self._ref_datasets_map = {
            k:v for k,v in zip(
                self.reference_names,
                split_dataset_random(self.parent_dataset, self.split_ratios, self.reference_names)
                )
            }

    def get_dataset_by_ref_name(self, ref_name: str)-> BaseDataset:
        return self._ref_datasets_map.get(ref_name)


def split_dataset_random(
        parent_dataset: ItemsDataset,
        fractions: collections.abc.Iterable[float],
        reference_names: collections.abc.Iterable[str]
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
            "reference_name": ref_name
        }
        # Create a new subdataset instance for the current subset of indices
        subdatasets.append(SubDataset(c_config))

    return subdatasets