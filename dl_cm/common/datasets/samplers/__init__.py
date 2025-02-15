from dl_cm.utils.registery import Registry
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.common import DLCM
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
from dl_cm.common.typing import namedEntitySchema
import torch

SAMPLER_REGISTRY = Registry("Samplers")
class BaseSampler(torch.utils.data.Sampler, DLCM, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        return namedEntitySchema

    def __init__(self, config):
        validationMixin.__init__(self, config)

class SamplersFactory(BaseFactory[BaseSampler]):
    
    @staticmethod
    def base_class()-> type[BaseSampler]:
        return BaseSampler

from functools import partial
base_sampler_adapter = partial(DLCM.base_class_adapter, base_cls=BaseSampler)

for cls in torch.utils.data.__dict__.values():
    if isinstance(cls, type) and issubclass(cls, torch.utils.data.Sampler):
        SAMPLER_REGISTRY.register(cls, base_class_adapter=base_sampler_adapter)

from . import hetero_dataset_batch_sampler 