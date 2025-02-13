
import glob, os
from dl_cm.common.datasets.filtered_dataset import ItemsDataset, FilteredItemsDataset
from dl_cm.common.datasets import CompositionDataset
from dl_cm.common.datasets import DATASETS_REGISTERY
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

    
    def __len__(self):
        return len(self.parent_dataset)
    
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
            self.IMAGE_KEY:self.read_image(item_fp).float()
        }
        return item
    