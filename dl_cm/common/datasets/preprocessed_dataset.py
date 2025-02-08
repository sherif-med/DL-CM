
from dl_cm.utils.registery import Registry
from typing import Union, Callable
from dl_cm.utils.exceptions import OutOfTypesException
from dl_cm.common.datasets import DATASETS_REGISTERY
from dl_cm.common.datasets import CompositionDataset
from dl_cm.common import DLCM
from typing import Type
from dl_cm.utils.ppattern.factory import BaseFactory

PREPROCESSING_REGISTERY = Registry("Preprocessing")

class PreprocessingBase(DLCM):
    @staticmethod
    def registry() -> Registry:
        return PREPROCESSING_REGISTERY

class PreprocessingFactory(BaseFactory):
    
    @staticmethod
    def base_class()-> Type["DLCM"]:
        return PreprocessingBase
    
@PREPROCESSING_REGISTERY.register(name="id")
class PreprocessingId(PreprocessingBase):
    def __init__(self):
        pass
    def __call__(self, item):
        return item


class PreprocessedDataset(CompositionDataset):
    
    def __init__(self, parent_dataset, preprocessing_fn: Union[str, dict, PreprocessingBase] = "id"):
        super().__init__(parent_dataset)
        self.preprocessing_callable = PreprocessingFactory.create(preprocessing_fn)
    
    def __len__(self):
        return len(self.parent_dataset)
    
    def __getitem__(self, index):
        return self.preprocessing_callable(self.parent_dataset[index])
