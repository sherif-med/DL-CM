import torch
import torch.utils.data.sampler

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


class SamplersFactory(BaseFactory[BaseSampler]):
    @staticmethod
    def base_class() -> type[BaseSampler]:
        return BaseSampler


# Register all samplers
for name in dir(torch.utils.data.sampler):
    cls = getattr(torch.utils.data.sampler, name)
    if isinstance(cls, type) and issubclass(cls, torch.utils.data.Sampler):
        _ = DLCM.base_class_adapter(cls, base_cls=BaseSampler)
