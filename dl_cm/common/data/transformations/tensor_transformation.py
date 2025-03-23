from .general_transformation import GeneralRevrsibleTransformation
from . import TRANSFORMATION_REGISTRY
from functools import partial
import numpy as np
import torch
from dl_cm.utils.exceptions import OutOfTypesException

class AlongPlaneTransformation:
    def __init__(self, spatial_dims):
        self.spatial_dims = spatial_dims

class AlongDimensionTransformation:
    def __init__(self, dimension_index):
        self.dimension_index = dimension_index


class Rotation2DTransformation(AlongPlaneTransformation):
    def __init__(self, spatial_dims=(-2, -1)):
        AlongPlaneTransformation.__init__(self, spatial_dims)
    
    def rot90_fn(item, times, axes=(-2,-1)):
        if isinstance(item, np.ndarray):
            return np.rot90(item, k=times, axes=axes)
        elif isinstance(item, torch.Tensor):
            return torch.rot90(item, k=times, dims=axes) 
        else:
            raise OutOfTypesException(item, (np.ndarray, torch.Tensor))

@TRANSFORMATION_REGISTRY.register()
class TransRot90(Rotation2DTransformation, GeneralRevrsibleTransformation):
    
    def __init__(self, spatial_dims=(-2, -1)):
        Rotation2DTransformation.__init__(self, spatial_dims)
        GeneralRevrsibleTransformation.__init__(self,
            fwdfn=partial(Rotation2DTransformation.rot90_fn, times=1),
            rwdfn=partial(Rotation2DTransformation.rot90_fn, times=3),
        )

@TRANSFORMATION_REGISTRY.register()
class TransRot180(Rotation2DTransformation, GeneralRevrsibleTransformation):
    
    def __init__(self, spatial_dims=(-2, -1)):
        Rotation2DTransformation.__init__(self, spatial_dims)
        GeneralRevrsibleTransformation.__init__(self,
            fwdfn=partial(Rotation2DTransformation.rot90_fn, times=2),
            rwdfn=partial(Rotation2DTransformation.rot90_fn, times=2),
        )

@TRANSFORMATION_REGISTRY.register()
class TransRot270(Rotation2DTransformation, GeneralRevrsibleTransformation):
    
    def __init__(self, spatial_dims=(-2, -1)):
        Rotation2DTransformation.__init__(self, spatial_dims)
        GeneralRevrsibleTransformation.__init__(self,
            fwdfn=partial(Rotation2DTransformation.rot90_fn, times=3),
            rwdfn=partial(Rotation2DTransformation.rot90_fn, times=1),
        )
    
@TRANSFORMATION_REGISTRY.register()
class Transflip(AlongDimensionTransformation, GeneralRevrsibleTransformation):
    
    def flip_fn(item, dimension_index):
        if isinstance(item, np.ndarray):
            return np.take(item, indices=np.arange(item.shape[dimension_index])[::-1], axis=dimension_index)
        elif isinstance(item, torch.Tensor):
            return torch.index_select( item, dimension_index, torch.arange(item.size(dimension_index)-1, -1, -1).to(item.device) )
        else:
            raise OutOfTypesException(item, (np.ndarray, torch.Tensor))
        
    def __init__(self, dimension_index):
        AlongDimensionTransformation.__init__(self, dimension_index)
        GeneralRevrsibleTransformation.__init__(self,
            fwdfn=partial(Transflip.flip_fn, dimension_index=self.dimension_index),
            rwdfn=partial(Transflip.flip_fn, dimension_index=self.dimension_index),
        )
