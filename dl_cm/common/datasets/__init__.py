import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property
from typing import Generic, Type, TypeVar

from pydantic import ConfigDict, validate_call

from dl_cm.common import DLCM
from dl_cm.common.typing import namedEntitySchema
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

DATASETS_REGISTERY = Registry("Datasets")
LOADED_DATASETS_REGISTRY = Registry("Loaded datasets")

COMPOSITION_DATASET_CLASS = TypeVar(
    "COMPOSITION_DATASET_CLASS_T", bound="CompositionDataset"
)


class BaseDataset(DLCM):
    _non_ref_datasets_counter = defaultdict(int)

    @staticmethod
    def registry() -> Registry:
        return DATASETS_REGISTERY

    @validate_call
    def __init__(
        self, reference_name: str = None, is_in_memory=False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._is_in_memory = is_in_memory
        if reference_name is None:
            reference_name = f"{self.__class__.__name__}_{BaseDataset._non_ref_datasets_counter[self.__class__]}"
            BaseDataset._non_ref_datasets_counter[self.__class__] += 1
        self._ref_name = reference_name
        if reference_name in LOADED_DATASETS_REGISTRY:
            raise ValueError(f"Dataset with name {reference_name} is already loaded")
        LOADED_DATASETS_REGISTRY.register(obj=self, name=self._ref_name)

    @property
    def reference_name(self) -> str:
        """
        Name of the dataset, used to differentiate between loaded datasets.

        This attribute is available only after the dataset is fully initialized.
        """
        return self._ref_name

    @property
    def is_in_memory(self) -> bool:
        return self._is_in_memory

    def compose(
        self, composition_cls: Type[COMPOSITION_DATASET_CLASS], *args, **kwargs
    ) -> COMPOSITION_DATASET_CLASS:
        """Compose a new dataset with the given composition class and kwargs.

        Args:
            composition_cls: Type of the composition class
            **kwargs: Additional arguments to pass to the composition class

        Returns:
            An instance of the composition class with the current dataset as the parent dataset
        """
        return composition_cls[COMPOSITION_DATASET_CLASS](
            parent_dataset=self, copy_parent=False, *args, **kwargs
        )


class DatasetFactory(BaseFactory[BaseDataset]):
    @staticmethod
    def base_class() -> Type["BaseDataset"]:
        return BaseDataset


COMPOSED_DATASET_CLASS = TypeVar("COMPOSED_DATASET_CLASS", bound=BaseDataset)
TOP_DATASET_CLASS = TypeVar("TOP_DATASET_CLASS", bound=BaseDataset)


class CompositionDataset(
    BaseDataset, ABC, Generic[COMPOSED_DATASET_CLASS, TOP_DATASET_CLASS]
):
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        parent_dataset: namedEntitySchema | COMPOSED_DATASET_CLASS,
        copy_parent: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        parent_dataset: COMPOSED_DATASET_CLASS = DatasetFactory.create(parent_dataset)
        self._parent_dataset: COMPOSED_DATASET_CLASS = (
            copy.copy(parent_dataset) if copy_parent else parent_dataset
        )

    @property
    def parent_dataset(self) -> COMPOSED_DATASET_CLASS:
        return self._parent_dataset

    @property
    def is_in_memory(self) -> bool:
        return self.parent_dataset.is_in_memory

    @abstractmethod
    def parent_index(self, index: int) -> int:
        """
        Abstract method to get the index in the parent dataset corresponding to the given index in the current dataset.

        Args:
            index (int): Index in the current dataset.

        Returns:
            int: Corresponding index in the parent dataset.
        """
        return index

    def top_parent_index(self, index):
        """
        Recursively retrieve the index in the top-level parent dataset
        corresponding to the given index in the current dataset.

        Args:
            index (int): Index in the current dataset.

        Returns:
            int: Corresponding index in the top-level parent dataset.
        """

        if not isinstance(self.parent_dataset, CompositionDataset):
            return self.parent_index(index)
        else:
            return self.parent_dataset.top_parent_index(self.parent_index(index))

    @cached_property
    def top_dataset(self) -> TOP_DATASET_CLASS:
        """
        Recursively retrieve the top-level parent dataset.

        Returns:
            BaseDataset: The top-level parent dataset.
        """
        if not isinstance(self.parent_dataset, CompositionDataset):
            return self.parent_dataset
        else:
            return self.parent_dataset.top_dataset


class IdentityCompositionDataset(
    CompositionDataset[COMPOSED_DATASET_CLASS, TOP_DATASET_CLASS]
):
    def parent_index(self, index: int) -> int:
        return index


from .augmented_dataset import AugmentedDataset
from .filtered_dataset import FilteredItemsDataset
from .items_dataset import ItemsDataset, UniqueItemsDataset
from .ordered_dataset import OrderedItemsDataset
from .preprocessed_dataset import PreprocessedDataset
from .shuffled_dataset import ShuffledDataset
from .split_datasets import SplitDataset
from .sub_dataset import SubDataset

__all__ = [
    "AugmentedDataset",
    "BaseDataset",
    "CompositionDataset",
    "FilteredItemsDataset",
    "ItemsDataset",
    "OrderedItemsDataset",
    "PreprocessedDataset",
    "ShuffledDataset",
    "SplitDataset",
    "SubDataset",
    "UniqueItemsDataset",
    "IdentityCompositionDataset",
]
