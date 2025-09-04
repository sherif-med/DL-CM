import numbers
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from skimage.io import imread
from torchvision.transforms import Resize

from dl_cm.common.data.datasets import BaseDataset
from dl_cm.common.data.datasets.filtered_dataset import FilteredItemsDataset
from dl_cm.common.data.datasets.images_dataset import (
    FilesWithinDirectoryDataset,
    ImagesWithinDirectoryDataset,
)
from dl_cm.common.data.datasets.items_dataset import CrossDataset
from dl_cm.common.data.transformations.general_transformation import (
    GeneralTransformation,
)
from dl_cm.common.data.transformations.multiple_items_transformation import (
    MultipleItemRevrsibleTransformation,
)
from dl_cm.common.data.transformations.tensor_transformation import (
    Transflip,
    TransRot90,
    TransRot180,
    TransRot270,
)
from dl_cm.common.typing import StepInputStruct


class VocDataset(BaseDataset):
    VOC_COLORMAP = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]

    def __init__(self, root_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        target_dir = os.path.join(root_dir, "SegmentationClass")
        images_dir = os.path.join(root_dir, "JPEGImages")
        self.images_path_dataset = FilesWithinDirectoryDataset(
            directory_path=images_dir
        ).compose(FilteredItemsDataset, ImagesWithinDirectoryDataset.is_image_fp)

        self.target_path_dataset = FilesWithinDirectoryDataset(
            directory_path=target_dir
        ).compose(FilteredItemsDataset, ImagesWithinDirectoryDataset.is_image_fp)

        self.cross_dataset = CrossDataset(
            datasets=[self.images_path_dataset, self.target_path_dataset],
            hash_fns=lambda x: Path(x).stem,
        )

    def __getitem__(self, index: numbers.Integral | slice):
        if isinstance(index, numbers.Integral):
            cross_items = self.cross_dataset[index]
            input_dict = {"image": torch.from_numpy(imread(cross_items[0])).float()}
            target_dict = {
                "label": self.convert_to_segmentation_mask(
                    torch.from_numpy(imread(cross_items[1]))
                )
            }
            return {"inputs": input_dict, "targets": target_dict}
        else:
            sub_indices = range(
                index.start, index.stop, index.step if index.step else 1
            )
            loaded_slice = [self.__getitem__(i) for i in sub_indices]
            return loaded_slice

    def convert_to_segmentation_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Converts a color segmentation mask to a multi-channel binary mask where each channel
        corresponds to a class as defined in self.color_map, using vectorized operations.

        Args:
            mask (torch.Tensor): Input mask of shape [height, width, channels].

        Returns:
            torch.Tensor: Segmentation mask of shape [height, width, num_classes].
        """
        # Create a tensor from the color map with shape [num_classes, channels]
        color_map_tensor = torch.tensor(
            self.VOC_COLORMAP, dtype=mask.dtype, device=mask.device
        )

        # Expand dimensions so that mask has shape [height, width, 1, channels]
        # and color_map_tensor has shape [1, 1, num_classes, channels]
        mask_expanded = mask.unsqueeze(2)  # shape: [H, W, 1, channels]
        color_map_expanded = color_map_tensor.unsqueeze(0).unsqueeze(
            0
        )  # shape: [1, 1, num_classes, channels]

        # Compare pixel colors with each label using broadcasting and then check all channels match
        segmentation_mask = (mask_expanded == color_map_expanded).all(dim=-1).float()
        # segmentation_mask now has shape: [height, width, num_classes]

        return segmentation_mask

    @staticmethod
    def plot_item(item: StepInputStruct, axs=None):
        if axs is None:
            _, axs = plt.subplots(1, 3, figsize=(20, 10))
        else:
            for ax in axs:
                ax.clear()
        axs[0].imshow(item["inputs"]["image"].int())
        axs[1].imshow(item["inputs"]["image"].int())
        axs[1].imshow(np.argmax(item["targets"]["label"], axis=-1), alpha=0.4)
        axs[2].imshow(np.argmax(item["targets"]["label"], axis=-1))
        plt.show()

    def __len__(self):
        return len(self.cross_dataset)


class VocPreprocessing(GeneralTransformation):
    @staticmethod
    def item_images_resize(item: StepInputStruct, resize_operator) -> StepInputStruct:
        item["inputs"]["image"] = resize_operator(item["inputs"]["image"].T)
        item["targets"]["label"] = resize_operator(item["targets"]["label"].T)
        return item

    def __init__(self, image_size=(256, 256)):
        resize_operator = Resize(image_size)
        super().__init__(
            partial(
                VocPreprocessing.item_images_resize, resize_operator=resize_operator
            )
        )


transformable_items_names = (
    "image",
    "label",
)


class VocTransRot90(MultipleItemRevrsibleTransformation):
    def __init__(
        self, spatial_dims=(-2, -1), transformable_items_names=transformable_items_names
    ):
        MultipleItemRevrsibleTransformation.__init__(
            self, TransRot90(spatial_dims), included_keys=transformable_items_names
        )


class VocTransRot180(MultipleItemRevrsibleTransformation):
    def __init__(
        self, spatial_dims=(-2, -1), transformable_items_names=transformable_items_names
    ):
        MultipleItemRevrsibleTransformation.__init__(
            self, TransRot180(spatial_dims), transformable_items_names
        )


class VocTransRot270(MultipleItemRevrsibleTransformation):
    def __init__(
        self, spatial_dims=(-2, -1), transformable_items_names=transformable_items_names
    ):
        MultipleItemRevrsibleTransformation.__init__(
            self, TransRot270(spatial_dims), transformable_items_names
        )


class VocTransFlipud(MultipleItemRevrsibleTransformation):
    def __init__(self, transformable_items_names=transformable_items_names):
        MultipleItemRevrsibleTransformation.__init__(
            self, Transflip(-2), transformable_items_names
        )


class VocTransFliplr(MultipleItemRevrsibleTransformation):
    def __init__(self, transformable_items_names=transformable_items_names):
        MultipleItemRevrsibleTransformation.__init__(
            self, Transflip(-1), transformable_items_names
        )


if __name__ == "__main__":
    dataset = VocDataset(root_dir="E:/Documents/DATA/VOC2012_train_val/")

    # Create figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    plt.subplots_adjust(bottom=0.25)  # Make room for the slider

    # Create slider axis
    slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
    index_slider = Slider(
        ax=slider_ax,
        label="Item Index",
        valmin=0,
        valmax=len(dataset) - 1,
        valinit=0,
        valstep=1,
    )

    # Update function for the slider
    def update(val):
        index = int(index_slider.val)
        item = dataset[index]

        dataset.plot_item(item, axs=axs)

        fig.canvas.draw_idle()

    # Register update function
    index_slider.on_changed(update)

    # Initial plot
    update(0)

    plt.show()
