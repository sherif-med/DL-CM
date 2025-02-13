import copy
from dl_cm.common.datasets import DATASETS_REGISTERY
from . import CompositionDataset
from collections.abc import Iterable
from dl_cm.common.datasets.transformations.general_transformation import GeneralRevrsibleTransformation

from .transformations import TRANSFORMATION_REGISTRY
from .transformations.general_transformation import GeneralTransformationFactory, GeneralTransformation
import pydantic as pd
from dl_cm.utils.ppattern.data_validation import validationMixin

@TRANSFORMATION_REGISTRY.register(name="id")
class TransIdentity(GeneralRevrsibleTransformation):
    def identity(x):
        return x
    
    def __init__(self):        
        GeneralRevrsibleTransformation.__init__(self,
            fwdfn=TransIdentity.identity,
            rwdfn=TransIdentity.identity,
        )


class AugmentedDataset(CompositionDataset, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            augmentations: list[str] = ["id"]
        return ValidConfig
    
    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        super().__init__(config)
        self.augmentations : list[GeneralTransformation] = \
            GeneralTransformationFactory.create(config.get("augmentations"))
    
    def __len__(self):
        return len(self.parent_dataset) * len(self.augmentations)
    
    def parent_index(self, index):
        parent_item_idx = index // (len(self.augmentations))
        return parent_item_idx
    
    def __getitem__(self, idx):        
        parent_item_idx = self.parent_index(idx)
        c_augmentation_idx = idx % (len(self.augmentations))
        c_augmentation = self.augmentations[c_augmentation_idx]
        
        parent_item = self.parent_dataset[parent_item_idx]
        return c_augmentation(parent_item)

