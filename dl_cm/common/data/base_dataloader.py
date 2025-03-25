import torch

from dl_cm.common.data.datasets import LOADED_DATASETS_REGISTRY
from dl_cm.common.data.samplers import SamplersFactory
from dl_cm.common.functions import FunctionsFactory
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

DATALOADERS_REGISTRY = Registry("Dataloaders")


class BaseDataloader:
    param_factory_map: dict[str, BaseFactory] = {
        "sampler": SamplersFactory,
        "batch_sampler": SamplersFactory,
        "collate_fn": FunctionsFactory,
        "pin_memory_device": FunctionsFactory,
    }

    def __init__(self, dataset_reference_name: str, *args, **kwargs):
        self.dataset_reference_name = dataset_reference_name

    @classmethod
    def instantiate_params(cls, params: dict) -> dict:
        """
        Instantiate all parameters that are factory-creatable

        Goes through all parameters and if a parameter name is in the
        param_factory_map, it uses the respective factory to instantiate the
        parameter.
        """
        for k, respective_factory in cls.param_factory_map.items():
            if k in params:
                params[k] = respective_factory.create(params.get(k))
        return params


def base_dataloader_adapter(dataloader_cls: type):
    """Decorator to make any external loss inherit from BaseLoss
    and adapt output to lossOutputStruct
    """

    class WrappedDataloader(dataloader_cls, BaseDataloader):
        def __init__(self: BaseDataloader, *args, **kwargs):
            """Wraps a loss to extract values from dictionary inputs."""
            BaseDataloader.__init__(self, *args, **kwargs)
            kwargs["dataset"] = LOADED_DATASETS_REGISTRY.get(
                self.dataset_reference_name
            )
            # instantiate object params
            kwargs = self.instantiate_params(kwargs)
            dataloader_cls.__init__(self, **kwargs)

    return WrappedDataloader


class DataloadersFactory(BaseFactory[BaseDataloader]):
    @staticmethod
    def base_class() -> type["BaseDataloader"]:
        return BaseDataloader


DATALOADERS_REGISTRY.register(
    torch.utils.data.DataLoader, base_class_adapter=base_dataloader_adapter
)
