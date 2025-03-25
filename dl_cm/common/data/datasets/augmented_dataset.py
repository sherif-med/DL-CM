from dl_cm.common.data.datasets import COMPOSED_DATASET_CLASS, TOP_DATASET_CLASS
from dl_cm.common.data.transformations import TRANSFORMATION_REGISTRY
from dl_cm.common.data.transformations.general_transformation import (
    GeneralRevrsibleTransformation,
    GeneralTransformation,
    GeneralTransformationFactory,
)

from . import CompositionDataset


@TRANSFORMATION_REGISTRY.register(name="id")
class TransIdentity(GeneralRevrsibleTransformation):
    @staticmethod
    def identity(x):
        return x

    def __init__(self):
        GeneralRevrsibleTransformation.__init__(
            self,
            fwdfn=TransIdentity.identity,
            rwdfn=TransIdentity.identity,
        )


class AugmentedDataset(CompositionDataset[COMPOSED_DATASET_CLASS, TOP_DATASET_CLASS]):
    def __init__(
        self, augmentations: list[str | GeneralTransformation], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.augmentations: list[GeneralTransformation] = (
            GeneralTransformationFactory.create(augmentations)
        )

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
