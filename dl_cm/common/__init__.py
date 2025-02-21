from dl_cm import _logger as logger
from dl_cm.utils.ppattern.self_registered_mixin import selfRegisteredMixin


class DLCM(selfRegisteredMixin):
    def __init__(self, *args, **kwargs):
        if kwargs:
            logger.warning(f"Unused kwargs: {kwargs} in {self.__class__.__name__}")

    @staticmethod
    def base_class_adapter(external_cls: type, base_cls: type) -> type:
        """Decorator to make any external class inherit from BaseClass"""

        class WrappedClass(external_cls, base_cls):
            def __init__(self, *args, **kwargs):
                external_cls.__init__(*args, **kwargs.pop("params", {}))
                base_cls.__init__(self, *args, **kwargs)

        return WrappedClass
