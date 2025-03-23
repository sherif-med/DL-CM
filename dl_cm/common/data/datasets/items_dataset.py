import collections
import collections.abc
from typing import Any, Callable

from dl_cm.common.datasets import (
    COMPOSED_DATASET_CLASS,
    TOP_DATASET_CLASS,
    DatasetFactory,
)
from dl_cm.common.datasets.sub_dataset import SubDataset
from dl_cm.common.functions import FunctionsFactory
from dl_cm.common.typing import OneOrMany, namedEntitySchema
from dl_cm.utils.exceptions import OutOfTypesException

from . import BaseDataset


class ItemsDataset(BaseDataset):
    """
    A dataset class that wraps a list of items, providing a standardized interface for working with in-memory datasets.

    This class is intended to differentiate between datasets that have already been loaded into memory and those that are lazy-loaded, such as generators.

    By using this class, you can ensure that your dataset is fully loaded and accessible, making it easier to work with libraries and frameworks that expect a fully-realized dataset.

    """

    def __init__(self, items: collections.abc.Iterable, *args, **kwargs):
        super().__init__(is_in_memory=True, *args, **kwargs)
        self._items_list = list(items)

    def __len__(self):
        return len(self._items_list)

    def __getitem__(self, index: int | slice) -> Any | list[Any]:
        return self._items_list[index]


class UniqueItemsDataset(SubDataset[COMPOSED_DATASET_CLASS, TOP_DATASET_CLASS]):
    """
    A dataset class that ensures all items within the dataset are unique based on a specified hash function.

    This class is useful for scenarios where you need to maintain a collection of unique items,
    and provides a standardized interface for working with such datasets. It extends from SubDataset
    and utilizes a hash function to identify and organize unique items within the parent dataset.
    """

    def __init__(
        self,
        parent_dataset: ItemsDataset | namedEntitySchema,
        hash_fn: str | Callable = hash,
        *args,
        **kwargs,
    ):
        self.hash_fn = FunctionsFactory.create(hash_fn)
        items_dataset = DatasetFactory.create(parent_dataset)
        if not items_dataset.is_in_memory:
            raise TypeError(
                f"Type mismatch. Expected ItemsDataset, got {type(items_dataset)}"
            )
        # create a set of unique items
        unique_items = {
            self.hash_fn(p_item): p_item for p_item in items_dataset
        }.values()
        # create a dict with unique items "hash" as keys and their respective indices as values
        self._index_map = {
            self.hash_fn(item): idx for idx, item in enumerate(unique_items)
        }
        # Initialize SubDataset
        sub_indices = [self._index_map[self.hash_fn(item)] for item in items_dataset]

        SubDataset.__init__(
            self, parent_dataset=items_dataset, indices=sub_indices, *args, **kwargs
        )

    def __contains__(self, item):
        return self.hash_fn(item) in self._index_map

    def index(self, item) -> int:
        return self._index_map[self.hash_fn(item)]


class CrossDataset(BaseDataset):
    """
    A dataset class that combines multiple datasets into a single cross-referenced dataset.

    This class ensures that all datasets are fully loaded and that each item can be cross-referenced
    across all provided datasets. It is useful for scenarios where you need to ensure that items
    from different datasets correspond to each other based on a specified hash function.

    Attributes:
        datasets: A list of UniqueItemsDataset instances created from the provided datasets.
        relative_indices: A list of tuples containing the indices of each item across all datasets.
    """

    def __init__(
        self,
        datasets: list[namedEntitySchema] | list[ItemsDataset],
        hash_fns: OneOrMany[str | Callable],
        strict_length: bool = False,  # if true, all datasets must have the same length
        strict_minimum_length: bool = False,  # if true, cross-dataset length must be at least the length of the shortest dataset
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        aux_datasets: list[ItemsDataset] = DatasetFactory.create(datasets)
        hash_fns = FunctionsFactory.create(hash_fns)
        if isinstance(hash_fns, Callable):
            hash_fns = [hash_fns] * len(aux_datasets)
        elif len(hash_fns) != len(aux_datasets):
            raise RuntimeError("Hash functions must have the same length as datasets!")
        self.datasets: list[UniqueItemsDataset] = []
        for dataset_idx, c_dataset in enumerate(aux_datasets):
            self.datasets.append(
                UniqueItemsDataset(c_dataset, hash_fn=hash_fns[dataset_idx])
            )
        if strict_length and len(set(map(len, self.datasets))) != 1:
            raise RuntimeError("Datasets must have the same length")
        self.relative_indices: list[tuple[int]] = []
        for item in self.datasets[0]:
            requested_indices = tuple(d.index(item) for d in self.datasets if item in d)
            if len(requested_indices) == len(self.datasets):
                self.relative_indices.append(requested_indices)
        if strict_minimum_length and len(self.relative_indices) < min(
            map(len, self.datasets)
        ):
            raise RuntimeError(
                "Cross-dataset length must be at least the length of the shortest dataset"
            )

    def __len__(self):
        return len(self.relative_indices)

    def __getitem__(self, index: int | slice) -> dict[str, Any] | dict[str, list[Any]]:
        if isinstance(index, int):
            cross_dataset_indices: tuple[int] | list[tuple[int]] = (
                self.relative_indices[index]
            )
            respective_datasets_items = [
                d[cross_dataset_indices[i]] for i, d in enumerate(self.datasets)
            ]
            return respective_datasets_items
        elif isinstance(index, slice):
            sub_indices = range(
                index.start, index.stop, index.step if index.step else 1
            )
            loaded_slice = [self.__getitem__(i) for i in sub_indices]
            return loaded_slice
        else:
            raise OutOfTypesException(index, (int, slice))
