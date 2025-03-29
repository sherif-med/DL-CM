from typing import Optional

import pytorch_lightning as pl

from dl_cm.common import DLCM
from dl_cm.common.data.base_dataloader import BaseDataloader, DataloadersFactory
from dl_cm.common.data.datasets import BaseDataset, DatasetFactory
from dl_cm.common.data.datasets.augmented_dataset import AugmentedDataset
from dl_cm.common.data.datasets.preprocessed_dataset import PreprocessedDataset
from dl_cm.common.data.datasets.split_datasets import SplitDataset
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
        loaded_datasets: list[BaseDataset] = DatasetFactory.create(datasets)
        self.datasets: dict[str, BaseDataset] = {}
        for c_dataset in loaded_datasets:
            # Add parts of splitDataset
            if isinstance(c_dataset, SplitDataset):
                self.datasets |= {
                    d_reference_name: c_dataset.get_dataset_by_ref_name(
                        d_reference_name
                    )
                    for d_reference_name in c_dataset.reference_names
                }
            # Add dataset as whole
            else:
                self.datasets[c_dataset.reference_name] = c_dataset

        # Data preprocessing
        preprocessing = FlaggedNamedEntity(**preprocessing)
        if preprocessing.apply:
            preprocessing_fn: GeneralTransformation = (
                GeneralTransformationFactory.create(preprocessing)
            )
            for c_dataset_ref_name, c_dataset in self.datasets.items():
                self.datasets[c_dataset_ref_name] = c_dataset.compose(
                    PreprocessedDataset, preprocessing_fn=preprocessing_fn
                )

        # Data augmentation
        if augmentation.get("apply", True):
            augmentations: list[GeneralTransformation] = (
                GeneralTransformationFactory.create(augmentation.get("augmentations"))
            )
            for c_dataset_ref_name, c_dataset in self.datasets.items():
                self.datasets[c_dataset_ref_name] = c_dataset.compose(
                    AugmentedDataset, augmentations=augmentations
                )

        # Dataloaders
        self.dataloaders: dict[str, BaseDataloader] = {}
        for dataloader_mode, dataloader_config in dataloaders.items():
            dataloader_config["params"] |= (
                common_dataloader_params if common_dataloader_params else {}
            )
            dataloader_config["params"]["dataset"] = self.datasets.get(
                dataloader_config["params"].pop("dataset_reference_name")
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
