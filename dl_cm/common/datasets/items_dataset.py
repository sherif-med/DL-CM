from typing import List, Any
from . import BaseDataset
import pydantic as pd
from dl_cm.utils.ppattern.data_validation import validationMixin

class ItemsDataset(BaseDataset, validationMixin):
    """
    A dataset class that wraps a list of items, providing a standardized interface for working with in-memory datasets.

    This class is intended to differentiate between datasets that have already been loaded into memory and those that are lazy-loaded, such as generators.

    By using this class, you can ensure that your dataset is fully loaded and accessible, making it easier to work with libraries and frameworks that expect a fully-realized dataset.

    Attributes:
        items (List[Any]): The list of items in the dataset
    """

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            items: list[Any]
        return ValidConfig
    
    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        super().__init__(config)
        items = config.get("items")
        self.items = items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
