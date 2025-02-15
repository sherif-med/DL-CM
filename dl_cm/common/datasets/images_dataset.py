
import glob
import os
from dl_cm.common.datasets.filtered_dataset import ItemsDataset, FilteredItemsDataset
from dl_cm.common.datasets import CompositionDataset
from skimage.io import imread
import torch
import pydantic as pd
from dl_cm.utils.ppattern.data_validation import validationMixin

class FilesWithinDirectoryDataset(ItemsDataset, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            directory_path: str
        return ValidConfig

    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        directory_path = config.get("directory_path")
        items_paths = glob.glob(os.path.join(directory_path, "*"))
        config["items"] = items_paths
        super().__init__(config)

DEFAULT_IMAGES_EXTENSIONS = tuple([".tif", ".jpeg", ".png"])
IMAGE_KEY = "image"

class ImagesWithinDirectoryDataset(CompositionDataset, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            directory_path: str
            image_extensions: list[str] = DEFAULT_IMAGES_EXTENSIONS
            image_key: str = IMAGE_KEY
        return ValidConfig

    def __init__(self, config: dict):
        validationMixin.__init__(self, config)
        super().__init__(config)
        self.image_key : str = config.get("image_key")
        def is_image_fp(filepath: str):
            return filepath.lower().endswith(config.get("image_extensions", DEFAULT_IMAGES_EXTENSIONS))
        root_directory_path = config.get("directory_path")
        filterd_dataset_config = {
            "filter_fn":is_image_fp,
            "parent_dataset":FilesWithinDirectoryDataset(root_directory_path)
        }
        config["parent_dataset"] = FilteredItemsDataset(filterd_dataset_config)
        config["copy_parent"] = False
        super().__init__(config)

    def __len__(self):
        return len(self.parent_dataset)

    def parent_index(self, index : int) -> int:
        return index

    @staticmethod
    def read_image(fp):
        loaded_tensor = torch.from_numpy(imread(fp))
        if len(loaded_tensor.shape)==2:
            # if 2D image, add band dimension as last dimension
            loaded_tensor = loaded_tensor.unsqueeze(-1)

        permutation_index = [i for i in range(2, len(loaded_tensor.shape))] + [0,  1]
        # Apply the permutation to the tensor
        reshaped_tensor = loaded_tensor.permute(*permutation_index)
        return reshaped_tensor

    def __getitem__(self, index):
        item_fp = self.parent_dataset[index]
        item = {
            "id":item_fp,
            self.image_key:self.read_image(item_fp).float()
        }
        return item
