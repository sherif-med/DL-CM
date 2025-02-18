import pytorch_lightning as pl
from pydantic import validate_call

from dl_cm.common import DLCM
from dl_cm.common.datasets import BaseDataset, DatasetFactory
from dl_cm.common.datasets.augmented_dataset import (
    AugmentedDataset,
    GeneralTransformation,
    GeneralTransformationFactory,
)
from dl_cm.common.datasets.base_dataloader import BaseDataloader, DataloadersFactory
from dl_cm.common.datasets.preprocessed_dataset import (
    BasePreprocessing,
    PreprocessedDataset,
    PreprocessingFactory,
)
from dl_cm.common.typing import namedEntitySchema
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

DATAMODULES_REGISTERY = Registry("DataModules")


class FlaggedNamedEntity(namedEntitySchema):
    apply: bool = True


class BaseDataModule(pl.LightningDataModule, DLCM):
    @validate_call
    def __init__(
        self,
        datasets: list[BaseDataset | namedEntitySchema],
        dataloaders: dict[str, namedEntitySchema | BaseDataloader],
        common_dataloader_params: dict = {},
        preprocessing=FlaggedNamedEntity(),
        augmentation=FlaggedNamedEntity(),
    ):
        super().__init__()
        # Datasets loading
        self.datasets: list[BaseDataset] = DatasetFactory.create(datasets)
        # Data preprocessing
        if preprocessing.apply:
            preprocessing_fn: BasePreprocessing = PreprocessingFactory.create(
                preprocessing.model_dump()
            )
            self.datasets = [
                d.compose(PreprocessedDataset, preprocessing_fn=preprocessing_fn)
                for d in self.datasets
            ]
        # Data augmentation
        if augmentation.apply:
            augmentations: list[GeneralTransformation] = (
                GeneralTransformationFactory.create(augmentation.model_dump())
            )
            self.datasets = [
                d.compose(AugmentedDataset, augmentations=augmentations)
                for d in self.datasets
            ]
        # Dataloaders
        self.dataloaders: dict[str, BaseDataloader] = {}
        for dataloader_mode, dataloader_config in dataloaders.items():
            dataloader_config["params"] |= common_dataloader_params
            self.dataloaders[dataloader_mode](
                DataloadersFactory.create(dataloader_config)
            )

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
