from .transformations.general_transformation import (
    TRANSFORMATION_REGISTRY,
    GeneralTransformation,
)


@TRANSFORMATION_REGISTRY.register(name="id")
class PreprocessingId(GeneralTransformation):
    def __init__(self):
        pass

    def __call__(self, item):
        return item
