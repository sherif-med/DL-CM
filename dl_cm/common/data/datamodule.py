from typing import Optional

import pytorch_lightning as pl

from dl_cm.common import DLCM
from dl_cm.common.data.base_dataloader import BaseDataloader, DataloadersFactory
from dl_cm.common.data.datasets import BaseDataset, DatasetFactory
from dl_cm.common.data.datasets.augmented_dataset import AugmentedDataset
from dl_cm.common.data.datasets.preprocessed_dataset import PreprocessedDataset
from dl_cm.common.data.transformations.general_transformation import (
    GeneralTransformation,
    GeneralTransformationFactory,
)
from dl_cm.common.typing import namedEntitySchema
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

DATAMODULES_REGISTERY = Registry("DataModules")


class FlaggedNamedEntity(namedEntitySchema):
    apply: bool = True


class BaseDataModule(pl.LightningDataModule, DLCM):
    @staticmethod
    def registry() -> Registry:
        return DATAMODULES_REGISTERY

    def __init__(
        self,
        datasets: list[BaseDataset | namedEntitySchema],
        dataloaders: dict[str, namedEntitySchema | BaseDataloader],
        preprocessing: dict,
        augmentation: dict,
        common_dataloader_params: Optional[dict] = None,
        extra: Optional[dict] = None,
    ):
        super().__init__()
        # Datasets loading
        self.datasets: list[BaseDataset] = DatasetFactory.create(datasets)
        # Data preprocessing
        preprocessing = FlaggedNamedEntity(**preprocessing)
        if preprocessing.apply:
            preprocessing_fn: GeneralTransformation = (
                GeneralTransformationFactory.create(preprocessing)
            )
            self.datasets = [
                d.compose(PreprocessedDataset, preprocessing_fn=preprocessing_fn)
                for d in self.datasets
            ]
        # Data augmentation
        if augmentation.get("apply", True):
            augmentations: list[GeneralTransformation] = (
                GeneralTransformationFactory.create(augmentation.get("augmentations"))
            )
            self.datasets = [
                d.compose(AugmentedDataset, augmentations=augmentations)
                for d in self.datasets
            ]
        # Dataloaders
        self.dataloaders: dict[str, BaseDataloader] = {}
        for dataloader_mode, dataloader_config in dataloaders.items():
            dataloader_config["params"] |= (
                common_dataloader_params if common_dataloader_params else {}
            )
            self.dataloaders[dataloader_mode] = DataloadersFactory.create(
                dataloader_config
            )
        self.extra_attributes = extra if extra else {}

    def train_dataloader(self):
        return self.dataloaders.get("train")

    def val_dataloader(self):
        return self.dataloaders.get("valid")

    def test_dataloader(self):
        return self.dataloaders.get("test")

    def predict_dataloader(self):
        return self.dataloaders.get("predict")


class DataModulesFactory(BaseFactory[BaseDataModule]):
    @staticmethod
    def base_class() -> type["BaseDataModule"]:
        return BaseDataModule
