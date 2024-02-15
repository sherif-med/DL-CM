
from dl_cm.utils.registery import Registry
from typing import Union, Callable

PREPROCESSING_REGISTERY = Registry("Preprocessing")

PREPROCESSING_REGISTERY.register(lambda x:x, "id")
PREPROCESSING_REGISTERY.register(lambda x:x, None)

class PreprocessedDataset:
    
    def __init__(self, parent_dataset, preprocessing_fn: Union[str, Callable] = "id"):
        super().__init__()
        self.parent_dataset = parent_dataset
        if isinstance(preprocessing_fn, str):
            preprocessing_fn = PREPROCESSING_REGISTERY.get(preprocessing_fn)
        self.preprocessing_callable = preprocessing_fn
    
    def __len__(self):
        return len(self.parent_dataset)
    
    def __getitem__(self, index):
        return self.preprocessing_callable(self.parent_dataset[index])
