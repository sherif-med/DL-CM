import torch
import torch.utils.data.sampler
from lightning.pytorch.overrides.distributed import _IndexBatchSamplerWrapper

from dl_cm.common import DLCM
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.registery import Registry

SAMPLER_REGISTRY = Registry("Samplers")


class BaseSampler(DLCM):
    @staticmethod
    def registry() -> Registry:
        return SAMPLER_REGISTRY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    param_factory_map: dict[str, BaseFactory] = {
        #"sampler": SamplersFactory,
    }
    
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


class SamplersFactory(BaseFactory[BaseSampler]):
    @staticmethod
    def base_class(similar=False) -> type[torch.utils.data.Sampler]:
        if similar:
            return (BaseSampler, torch.utils.data.Sampler, _IndexBatchSamplerWrapper)
        return BaseSampler

def base_sampler_adapter(sampler_cls: type):
    """Decorator to make any external sampler inherit from BaseSampler
    """

    class WrappedSampler(sampler_cls, BaseSampler):
        def __init__(self: BaseSampler, *args, **kwargs):
            """Wraps a sampler to pop unused arguments and instantiate params."""
            BaseSampler.__init__(self)
            # instantiate object params
            kwargs = self.instantiate_params(kwargs)
            sampler_cls.__init__(self, **kwargs)

    return WrappedSampler

# Register all samplers
for name in dir(torch.utils.data.sampler):
    attr = getattr(torch.utils.data.sampler, name)
    if isinstance(attr, type) and issubclass(attr, torch.utils.data.Sampler):
        SAMPLER_REGISTRY.register(
            obj=attr, name=name, base_class_adapter=base_sampler_adapter
        )
