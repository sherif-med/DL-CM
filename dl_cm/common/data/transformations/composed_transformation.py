from .general_transformation import GeneralTransformation, GeneralTransformationFactory
from . import TRANSFORMATION_REGISTRY

@TRANSFORMATION_REGISTRY.register()
class ComposedTransformation(GeneralTransformation):
    
    def __init__(self, *sub_transforms):
        self.sub_transforms = list()
        for c_transform in sub_transforms:
            self.sub_transforms.append(GeneralTransformationFactory.create(c_transform))
        
    
    def __fwd__(self, item, **kwargs):
        out_item = item
        for sub_t in self.sub_transforms:
            out_item = sub_t(out_item, **kwargs)
        return out_item
    
    def __rwd__(self, item, **kwargs):
        out_item = item
        for sub_t in self.sub_transforms[::-1]:
            out_item = sub_t(out_item, reverse=True, **kwargs)
        return out_item