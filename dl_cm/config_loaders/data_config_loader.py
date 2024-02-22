from dl_cm.common.datasets import DATASETS_REGISTERY
from dl_cm.common.datasets.augmented_dataset import AugmentedDataset
from dl_cm.common.datasets.preprocessed_dataset import PreprocessedDataset, PREPROCESSING_REGISTERY
from dl_cm.common.datasets.split_datasets import split_subdatasets_random
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dl_cm import _logger
from functools import partial
from dl_cm.common.datasets.datamodule import PartialDataModule
from dl_cm.config_loaders import load_named_entity

default_augmentations = set(("id",))

def datasets_from_config(datainfo_config:dict):
    """
    Returns a map of loaded datasets using datainfo section
    """
    
    augmentation_switch = datainfo_config.get("apply_augmentation", True)
    
    loaded_preprocessing = load_named_entity(PREPROCESSING_REGISTERY, datainfo_config.get("preprocessing"))
    
    datasets_map = {}
    
    for c_dataset_info in datainfo_config.get("datasets"):
        # Iterate through every dataset definitions
        c_name = c_dataset_info.get("name")
        
        loaded_dataset = load_named_entity(DATASETS_REGISTERY, c_dataset_info.get("datasettype"))
        
        # Apply preprocessing
        preprocessed_dataset = PreprocessedDataset(loaded_dataset, loaded_preprocessing)
        
        # Apply augmentation
        c_augmentations = set(c_dataset_info.get("augmentations", default_augmentations))
        if not augmentation_switch and c_augmentations != default_augmentations:
            _logger.warn(f"Global Augmentation switch 'apply_augmentation' is off! \
                        Augmentation for dataset with name: {c_name} will be ignored!")
            c_augmentations = default_augmentations
        
        augmented_dataset = AugmentedDataset(preprocessed_dataset, c_augmentations)
        
        # If '/' is present in the name, the current dataset should be split into many for example 'train/val/test'
        parts = c_name.split("/")
        assert all([p not in datasets_map for p in parts]), "Some dataset are defined twice with the same name!"
        if len(parts)==1:
            datasets_map[c_name] = augmented_dataset
        else:
            subdatasets_fractions = c_dataset_info.get("split_ratio", (1,))
            assert len(parts)==len(subdatasets_fractions), "Split ratios don't correspond to the number of subdatasets!"
            subdatasets = split_subdatasets_random(augmented_dataset, subdatasets_fractions)
            datasets_map.update({k: v for k, v in zip(parts, subdatasets)})
    
    return datasets_map
        

def datamodule_from_config(datainfo_config:dict)->pl.LightningDataModule:
    """
    Returns a map of dataloaders using datainfo section
    """
    datasets_map = datasets_from_config(datainfo_config)
    
    common_loaders_params = datainfo_config.get("common_loaders_params", {})
    
    partial_dataloaders_map = {}
    
    for c_dataloader_info in datainfo_config.get("dataloaders"):
        c_loader_params = dict(common_loaders_params)
        c_loader_params.update(c_dataloader_info.get("params"))
        partial_dataloaders_map[c_dataloader_info.get("name")] = partial(DataLoader,
            datasets_map[c_dataloader_info.get("respective_dataset_name")],
            **c_loader_params
            )
    
    loaded_datamodule = PartialDataModule(**partial_dataloaders_map)
    return loaded_datamodule    