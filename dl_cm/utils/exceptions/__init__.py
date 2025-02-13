
class OutOfTypesException(Exception):
    def __init__(self, obj, allowed_type_tuple):
        super(OutOfTypesException, self).__init__(f"Instance {obj} of type {type(obj)} is out of allowed types: {list(allowed_type_tuple)}")

class BaseClassNotInitializedException(Exception):
    def __init__(self, obj):
        super(BaseClassNotInitializedException, self).__init__(f"Base class of Instance {obj} of type {type(obj)} is not initialized")
