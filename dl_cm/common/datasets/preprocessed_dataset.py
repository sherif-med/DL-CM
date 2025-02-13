
from dl_cm.utils.registery import Registry
from typing import Union, Callable
from dl_cm.utils.exceptions import OutOfTypesException
from dl_cm.common.datasets import DATASETS_REGISTERY
from dl_cm.common.datasets import CompositionDataset
from dl_cm.common import DLCM
from typing import Type
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd

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


class PreprocessedDataset(CompositionDataset, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            preprocessing_fn: str | Callable = "id"
        return ValidConfig
    
    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        super().__init__(config)
        self.preprocessing_callable = PreprocessingFactory.create(config.get("preprocessing_fn"))
    
    def __len__(self):
        return len(self.parent_dataset)
    
    def __getitem__(self, index):
        return self.preprocessing_callable(self.parent_dataset[index])
