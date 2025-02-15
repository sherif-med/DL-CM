from typing import TypeVar, Generic, Type
from dl_cm.utils.registery import Registry
from dl_cm.utils.ppattern.factory import BaseFactory
import copy
from dl_cm.common import DLCM
from dl_cm.utils.ppattern.data_validation import validationMixin
from dl_cm.utils.ppattern.init_check_mixin import InitCheckMixin
from dl_cm.common.typing import namedEntitySchema
import pydantic as pd
from abc import ABC, abstractmethod

DATASETS_REGISTERY = Registry("Datasets")
LOADED_DATASETS_REGISTRY = Registry("Loaded datasets")

class BaseDataset(DLCM, validationMixin, InitCheckMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(namedEntitySchema):
            reference_name: str = None # this field is used to differentiate between loaded datasets
        return ValidConfig

    @staticmethod
    def registry() -> Registry:
        return DATASETS_REGISTERY

    def __init__(self, config: dict) -> None:
        validationMixin.__init__(self, config)
        InitCheckMixin.__init__(self)
        ref_name = config.get("reference_name", self.__class__.__name__)
        if ref_name in LOADED_DATASETS_REGISTRY:
            raise ValueError(f"Dataset with name {ref_name} is already loaded")
        LOADED_DATASETS_REGISTRY.register(obj=self, name=ref_name)
        self._ref_name = ref_name

    @property
    def reference_name(self) -> str:
        """
        Name of the dataset, used to differentiate between loaded datasets.

        This attribute is available only after the dataset is fully initialized.
        """
        self.check_base_class_initialized()
        return self._ref_name

    def compose(self, composition_cls: Type["CompositionDataset"], **kwargs) -> "CompositionDataset":
        """Compose a new dataset with the given composition class and kwargs.

        Args:
            composition_cls: Type of the composition class
            **kwargs: Additional arguments to pass to the composition class

        Returns:
            An instance of the composition class with the current dataset as the parent dataset
        """
        return composition_cls({"parent_dataset": self, "copy_parent": False, **kwargs})

class DatasetFactory(BaseFactory[BaseDataset]):

    @staticmethod
    def base_class()-> Type["BaseDataset"]:
        return BaseDataset

COMPOSED_DATASET_CLASS = TypeVar('COMPOSED_CLASS', bound=BaseDataset)
class CompositionDataset(BaseDataset, validationMixin, InitCheckMixin, ABC, Generic[COMPOSED_DATASET_CLASS]):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            parent_dataset: namedEntitySchema | BaseDataset
            copy_parent: bool = False
        return ValidConfig

    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        InitCheckMixin.__init__(self)
        super().__init__(config)
        parent_dataset : COMPOSED_DATASET_CLASS = DatasetFactory.create(config.get("parent_dataset"))
        self._parent_dataset : COMPOSED_DATASET_CLASS = copy.copy(parent_dataset) if config.get("copy_parent") else parent_dataset

    @property
    def parent_dataset(self) -> COMPOSED_DATASET_CLASS:
        self.check_base_class_initialized()
        return self._parent_dataset

    @abstractmethod
    def parent_index(self, index : int) -> int:
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

    def get_top_dataset(self):
        """
        Recursively retrieve the top-level parent dataset.

        Returns:
            BaseDataset: The top-level parent dataset.
        """
        if not isinstance(self.parent_dataset, CompositionDataset):
            return self.parent_dataset
        else:
            return self.parent_dataset.get_top_dataset()

from . import combined_dataset
from . import shuffled_dataset