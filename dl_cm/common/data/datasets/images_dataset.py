import collections
import glob
import os
from functools import partial

import torch
from pydantic import DirectoryPath, validate_call
from skimage.io import imread

from dl_cm.common.data.datasets import (
    COMPOSED_DATASET_CLASS,
    TOP_DATASET_CLASS,
    CompositionDataset,
)
from dl_cm.common.data.datasets.filtered_dataset import FilteredItemsDataset
from dl_cm.common.data.datasets.items_dataset import ItemsDataset


class FilesWithinDirectoryDataset(ItemsDataset):
    @validate_call
    def __init__(self, directory_path: DirectoryPath, *args, **kwargs):
        items_paths = glob.glob(os.path.join(directory_path, "*"))
        super().__init__(items=items_paths, *args, **kwargs)


DEFAULT_IMAGES_EXTENSIONS = tuple([".tif", ".jpeg", ".png", "jpg"])
IMAGE_KEY = "image"


class ImagesWithinDirectoryDataset(
    CompositionDataset[COMPOSED_DATASET_CLASS, TOP_DATASET_CLASS]
):
    def __init__(
        self,
        directory_path: DirectoryPath,
        image_extensions: collections.abc.Iterable[str] = DEFAULT_IMAGES_EXTENSIONS,
        image_key: str = IMAGE_KEY,  # used as key for image tensor within getitem dict
        *args,
        **kwargs,
    ):
        self.image_key = image_key

        parent_dataset = FilteredItemsDataset(
            filter_fn=partial(
                ImagesWithinDirectoryDataset.is_image_fp,
                image_extensions=image_extensions,
            ),
            parent_dataset=FilesWithinDirectoryDataset(directory_path=directory_path),
        )
        super().__init__(parent_dataset=parent_dataset, *args, **kwargs)

    @staticmethod
    def is_image_fp(
        filepath: str,
        image_extensions: collections.abc.Iterable[str] = DEFAULT_IMAGES_EXTENSIONS,
    ) -> bool:
        return filepath.lower().endswith(image_extensions)

    def __len__(self):
        return len(self.parent_dataset)

    def parent_index(self, index: int) -> int:
        return index

    @staticmethod
    def read_image(fp):
        loaded_tensor = torch.from_numpy(imread(fp))
        if len(loaded_tensor.shape) == 2:
            # if 2D image, add band dimension as last dimension
            loaded_tensor = loaded_tensor.unsqueeze(-1)

        permutation_index = [i for i in range(2, len(loaded_tensor.shape))] + [0, 1]
        # Apply the permutation to the tensor
        reshaped_tensor = loaded_tensor.permute(*permutation_index)
        return reshaped_tensor

    def __getitem__(self, index):
        item_fp = self.parent_dataset[index]
        item = {"id": item_fp, self.image_key: self.read_image(item_fp).float()}
        return item
