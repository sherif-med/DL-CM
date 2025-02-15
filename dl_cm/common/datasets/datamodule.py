from dl_cm.common.typing import namedEntitySchema
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
from dl_cm.utils.registery import Registry
from dl_cm.common.datasets import DatasetFactory, BaseDataset
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.common import DLCM
from dl_cm.common.datasets.base_dataloader import DataloadersFactory, BaseDataloader
from dl_cm.common.datasets.preprocessed_dataset import PreprocessingFactory, PreprocessedDataset, BasePreprocessing
from dl_cm.common.datasets.augmented_dataset import AugmentedDataset, GeneralTransformation, GeneralTransformationFactory
import pytorch_lightning as pl

DATAMODULES_REGISTERY = Registry("DataModules")

class BaseDataModule(pl.LightningDataModule, DLCM, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class FlaggedNamedEntity(namedEntitySchema):
            apply: bool = True
        class ValidConfig(namedEntitySchema):
            datasets: dict[str, namedEntitySchema]
            dataloaders: dict[str, namedEntitySchema]
            common_dataloader_params: dict = {}
            preprocessing: FlaggedNamedEntity = {}
            augmentation: FlaggedNamedEntity = {}
        return ValidConfig

    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        super().__init__()
        # Datasets loading
        self.datasets : list[BaseDataset] = DatasetFactory.create(config.get("datasets"))
        # Data preprocessing
        pre_processing_config : dict = config.get("preprocessing")
        if pre_processing_config and pre_processing_config.pop("apply"):
            pre_processing : BasePreprocessing = PreprocessingFactory.create(pre_processing_config)
            self.datasets = [d.compose(PreprocessedDataset, preprocessing_fn=pre_processing) for d in self.datasets]
        # Data augmentation
        augmentation_config :dict = config.get("augmentation")
        if augmentation_config and augmentation_config.pop("apply"):
            augmentations : list[GeneralTransformation] = GeneralTransformationFactory.create(augmentation_config.get("augmentations"))
            self.datasets = [d.compose(AugmentedDataset, augmentations=augmentations) for d in self.datasets]
        # Dataloaders
        self.dataloaders : dict[str,BaseDataloader] = {}
        for dataloader_mode, dataloader_config in config.get("dataloaders").items():
            dataloader_config["params"] |= config.get("common_dataloader_params")
            self.dataloaders[dataloader_mode](DataloadersFactory.create(dataloader_config))
    
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
    def base_class()-> type["BaseDataModule"]:
        return BaseDataModule