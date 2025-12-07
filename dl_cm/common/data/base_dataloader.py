import torch

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

    def __init__(self, *args, **kwargs):
        pass

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

    @staticmethod
    def registry() -> Registry:
        return DATALOADERS_REGISTRY


def base_dataloader_adapter(dataloader_cls: type):
    """Decorator to make any external dataloader inherit from BaseDataloader
    """

    class WrappedDataloader(dataloader_cls, BaseDataloader):
        def __init__(self: BaseDataloader, *args, **kwargs):
            """Wraps a data loader to pop unused arguments and instantiate params."""
            BaseDataloader.__init__(self)
            # instantiate object params
            kwargs = self.instantiate_params(kwargs)
            dataloader_cls.__init__(self, **kwargs)

    return WrappedDataloader


class DataloadersFactory(BaseFactory[BaseDataloader]):
    @staticmethod
    def base_class(similar=False) -> type["BaseDataloader"]:
        if similar:
            return (BaseDataloader,)
        return BaseDataloader


DATALOADERS_REGISTRY.register(
    torch.utils.data.DataLoader, base_class_adapter=base_dataloader_adapter
)
