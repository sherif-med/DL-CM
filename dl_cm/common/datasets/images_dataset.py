
import glob, os
from dl_cm.common.datasets.filtered_dataset import ItemsDataset, FilteredItemsDataset
from dl_cm.common.datasets import CompositionDataset
from dl_cm.common.datasets import DATASETS_REGISTERY
from skimage.io import imread
import torch

class FilesWithinDirectoryDataset(ItemsDataset):
    
    def __init__(self, root_directory_path):        
        self.root_directory_path = root_directory_path
        items_paths = glob.glob(os.path.join(root_directory_path, "*"))
        ItemsDataset.__init__(self, items_paths)


@DATASETS_REGISTERY.register()
class ImagesWithinDirectoryDataset(CompositionDataset):
    
    DEFAULT_IMAGES_EXTENSIONS = tuple([".tif", ".jpeg", ".png"])
    IMAGE_KEY = "image"
    
    def __init__(self, root_directory_path, images_extension=DEFAULT_IMAGES_EXTENSIONS):
        """
        Args:
            root_directory_path (str).
            images_extension (List(str), optional). Defaults to DEFAULT_IMAGES_EXTENSIONS.
        """
        
        is_image_fp = lambda fp:fp.lower().endswith(images_extension)
        super().__init__(FilteredItemsDataset(FilesWithinDirectoryDataset(root_directory_path), is_image_fp), copy_parent=False)
    
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
            self.IMAGE_KEY:self.read_image(item_fp)
        }
        return item
    