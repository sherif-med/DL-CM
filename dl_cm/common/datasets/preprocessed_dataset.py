from typing import Callable, Type

from dl_cm.common import DLCM
from dl_cm.common.datasets import COMPOSED_DATASET_CLASS, CompositionDataset
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

PREPROCESSING_REGISTERY = Registry("Preprocessing")


class BasePreprocessing(DLCM):
    @staticmethod
    def registry() -> Registry:
        return PREPROCESSING_REGISTERY


class PreprocessingFactory(BaseFactory[BasePreprocessing]):
    @staticmethod
    def base_class() -> Type["DLCM"]:
        return BasePreprocessing


@PREPROCESSING_REGISTERY.register(name="id")
class PreprocessingId(BasePreprocessing):
    def __init__(self):
        pass

    def __call__(self, item):
        return item


class PreprocessedDataset(CompositionDataset[COMPOSED_DATASET_CLASS]):
    def __init__(self, preprocessing_fn: str | Callable = "id", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessing_callable = PreprocessingFactory.create(preprocessing_fn)

    def __len__(self):
        return len(self.parent_dataset)

    def __getitem__(self, index):
        return self.preprocessing_callable(self.parent_dataset[index])
