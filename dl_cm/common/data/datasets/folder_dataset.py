import os, glob
from pathlib import Path
from typing import Callable

from dl_cm.common.data.datasets import CompositionDataset, COMPOSED_DATASET_CLASS, TOP_DATASET_CLASS
from dl_cm.common.data.datasets import ItemsDataset, FilteredItemsDataset
from dl_cm.common.functions import FunctionsFactory

class ListDirectoryDataset(ItemsDataset):
    def __init__(self, directory_path: Path, *args, **kwargs):
        assert os.path.isdir(directory_path), "Folder not found"
        items_paths = glob.glob(os.path.join(directory_path, "*"))
        #items_paths = [Path(i) for i in items_paths]
        super().__init__(items=items_paths, *args, **kwargs)

class FilesWithinDirectoryDataset(
    CompositionDataset[COMPOSED_DATASET_CLASS, TOP_DATASET_CLASS]
):
    def __init__(
        self,
        directory_path: Path,
        extension_loader_map: dict[str | frozenset, Callable | str], # A dict of relative file loader to every extension
        extension_key_map: dict = {}, # a dict of allowed extension and relative key in item dict
        *args,
        **kwargs,
    ):
        self.extensions_lodaer_map = {k: FunctionsFactory.create(v) for k,v in extension_loader_map.items()}
        self.extension_key_map = extension_key_map
        parent_dataset = FilteredItemsDataset(
            filter_fn=lambda x: Path(x).suffix.lower() in extension_loader_map.keys(),
            parent_dataset=ListDirectoryDataset(directory_path=directory_path),
        )
        super().__init__(parent_dataset=parent_dataset, *args, **kwargs)

    def __len__(self):
        return len(self.parent_dataset)

    def parent_index(self, index: int) -> int:
        return index

    def __getitem__(self, index):
        item_fp = self.parent_dataset[index]
        item_extension = Path(item_fp).suffix
        extension_relative_key = self.extension_key_map.get(item_extension, item_extension) # TODO: add test for frozenset option. Eg: (png, jpg,...) 
        extension_file_loader = self.extensions_lodaer_map.get(item_extension)
        item = {"id": item_fp, extension_relative_key: extension_file_loader(item_fp)}
        return item
