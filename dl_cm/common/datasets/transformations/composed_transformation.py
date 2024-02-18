from .general_transformation import GeneralRevrsibleTransformation

class ComposedTransformation(GeneralRevrsibleTransformation):
    
    def __init__(self, *sub_transforms):
        self.sub_transforms=list(sub_transforms)
    
    def __fwd__(self, item, **kwargs):
        out_item = item
        for sub_t in self.sub_transforms:
            out_item = sub_t(out_item, **kwargs)
        return out_item
    
    def __rwd__(self, item, **kwargs):
        out_item = item
        for sub_t in self.sub_transforms[::-1]:
            out_item = sub_t(out_item, **kwargs)
        return out_item