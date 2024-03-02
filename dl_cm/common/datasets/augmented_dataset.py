import copy
from dl_cm.utils.registery import Registry
from . import CompositionDataset
from collections.abc import Iterable
from dl_cm.common.datasets.transformations.general_transformation import GeneralRevrsibleTransformation

from .transformations import TRANSFORMATION_REGISTRY

@TRANSFORMATION_REGISTRY.register(name="id")
class TransIdentity(GeneralRevrsibleTransformation):
    def identity(x):
        return x
    
    def __init__(self):        
        GeneralRevrsibleTransformation.__init__(self,
            fwdfn=TransIdentity.identity,
            rwdfn=TransIdentity.identity,
        )

class AugmentedDataset(CompositionDataset):
    
    def __init__(self, parent_dataset, augmentations:Iterable=("id",)):
        CompositionDataset.__init__(self, parent_dataset)
        if isinstance(next(iter(augmentations)), str):
            augmentations = [TRANSFORMATION_REGISTRY.get(k)() for k in augmentations]
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.parent_dataset) * len(self.augmentations)
    
    def __getitem__(self, idx):        
        parent_item_idx = idx // (len(self.augmentations))
        c_augmentation_idx = idx % (len(self.augmentations))
        c_augmentation = self.augmentations[c_augmentation_idx]
        
        parent_item = self.parent_dataset[parent_item_idx]
        return c_augmentation(parent_item)

