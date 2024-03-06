import copy
from dl_cm.common.datasets import DATASETS_REGISTERY
from . import CompositionDataset
from collections.abc import Iterable
from dl_cm.common.datasets.transformations.general_transformation import GeneralRevrsibleTransformation

from .transformations import TRANSFORMATION_REGISTRY
from .transformations.general_transformation import GeneralTransformationFactory

@TRANSFORMATION_REGISTRY.register(name="id")
class TransIdentity(GeneralRevrsibleTransformation):
    def identity(x):
        return x
    
    def __init__(self):        
        GeneralRevrsibleTransformation.__init__(self,
            fwdfn=TransIdentity.identity,
            rwdfn=TransIdentity.identity,
        )

@DATASETS_REGISTERY.register()
class AugmentedDataset(CompositionDataset):
    
    def __init__(self, parent_dataset, augmentations:Iterable=("id",)):
        CompositionDataset.__init__(self, parent_dataset)
        self.augmentations = [GeneralTransformationFactory.create(k) for k in augmentations]
    
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

