
from dl_cm.utils.registery import Registry
from typing import Union, Callable
from dl_cm.utils.exceptions import OutOfTypesException

PREPROCESSING_REGISTERY = Registry("Preprocessing")

class PreprocessingBase:
    pass

@PREPROCESSING_REGISTERY.register(name="id")
class PreprocessingId(PreprocessingBase):
    def __init__(self):
        pass
    def __call__(self, item):
        return item


class PreprocessedDataset:
    
    def __init__(self, parent_dataset, preprocessing_fn: Union[str, PreprocessingBase] = "id"):
        super().__init__()
        self.parent_dataset = parent_dataset
        if isinstance(preprocessing_fn, str):
            preprocessing_fn = PREPROCESSING_REGISTERY.get(preprocessing_fn)
        elif isinstance(preprocessing_fn, PreprocessingBase):
            self.preprocessing_callable = preprocessing_fn
        else:
            raise OutOfTypesException(preprocessing_fn, (str, PreprocessingBase))
    
    def __len__(self):
        return len(self.parent_dataset)
    
    def __getitem__(self, index):
        return self.preprocessing_callable(self.parent_dataset[index])
