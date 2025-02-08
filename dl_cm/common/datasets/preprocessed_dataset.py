
from dl_cm.utils.registery import Registry
from typing import Union, Callable
from dl_cm.utils.exceptions import OutOfTypesException
from dl_cm.common.datasets import DATASETS_REGISTERY
from dl_cm.common.datasets import CompositionDataset

PREPROCESSING_REGISTERY = Registry("Preprocessing")

class PreprocessingBase:
    pass

class PreprocessingFactory:
    def __init__(self):
        pass

    @classmethod
    def create(cls, c_preprocessing):
        if isinstance(c_preprocessing, str):
            preprocessing_class = PREPROCESSING_REGISTERY.get(c_preprocessing)
            return preprocessing_class()
        elif isinstance(c_preprocessing, dict):
            return load_named_entity(PREPROCESSING_REGISTERY, c_preprocessing)
        elif isinstance(c_preprocessing, PreprocessingBase):
            return c_preprocessing
        else:
            raise OutOfTypesException(c_preprocessing, (str, dict, PreprocessingBase,))


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
