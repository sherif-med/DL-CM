import copy
from src.utils.registery import Registry
from . import CompositionDataset

AUGMENTATION_TRANSFORMATION_REGISTERY = Registry("Augmentation transformation")

class AugmentedDataset(CompositionDataset):
    
    def __init__(self, parent_dataset, augmentations=("id",)):
        CompositionDataset.__init__(self, parent_dataset)
        if isinstance(augmentations[0], str):
            augmentations = [AUGMENTATION_TRANSFORMATION_REGISTERY.get(k) for k in augmentations]
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.parent_dataset) * len(self.augmentations)
    
    def __getitem__(self, idx):        
        parent_item_idx = idx // (len(self.augmentations))
        c_augmentation_idx = idx % (len(self.augmentations))
        c_augmentation = self.augmentations[c_augmentation_idx]
        
        parent_item = self.parent_dataset[parent_item_idx]
        return c_augmentation(parent_item)

