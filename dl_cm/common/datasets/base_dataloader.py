import pydantic as pd
import torch

from dl_cm.common.datasets import LOADED_DATASETS_REGISTRY
from dl_cm.common.datasets.samplers import SamplersFactory
from dl_cm.common.functions import FunctionsFactory
from dl_cm.common.typing import namedEntitySchema
from dl_cm.utils.ppattern.data_validation import validationMixin
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

DATALOADERS_REGISTRY = Registry("Dataloaders")


def base_dataloader_adapter(dataloader_cls: type):
    """Decorator to make any external loss inherit from BaseLoss
    and adapt output to lossOutputStruct
    """

    class WrappedDataloader(dataloader_cls, BaseDataloader):
        def __init__(self, config: namedEntitySchema):
            """Wraps a loss to extract values from dictionary inputs."""
            BaseDataloader.__init__(self, config)
            config.params["dataset"] = LOADED_DATASETS_REGISTRY.get(
                config["dataset_reference_name"]
            )
            # instantiate object params
            config.params = self.instantiate_params(config.params)
            super().__init__(config.params)

    return WrappedDataloader


class BaseDataloader(validationMixin):
    param_factory_map: dict[str, BaseFactory] = {
        "sampler": SamplersFactory,
        "batch_sampler": SamplersFactory,
        "collate_fn": FunctionsFactory,
        "pin_memory_device": FunctionsFactory,
    }

    @staticmethod
    def config_schema() -> pd.BaseModel:
        class ValidConfig(namedEntitySchema):
            dataset_reference_name: str

        return ValidConfig

    def __init__(self, config: namedEntitySchema):
        validationMixin.__init__(self, config)

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


class DataloadersFactory(BaseFactory[BaseDataloader]):
    @staticmethod
    def base_class() -> type["BaseDataloader"]:
        return BaseDataloader


DATALOADERS_REGISTRY.register(
    torch.utils.data.DataLoader, base_class_adapter=base_dataloader_adapter
)
