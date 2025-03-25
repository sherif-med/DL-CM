from dl_cm import _logger as logger
from dl_cm.utils.registery import registeredClassMixin


class DLCM(registeredClassMixin):
    def __init_subclass__(cls, *args, wrapped_name=None, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if not wrapped_name:
            wrapped_name = cls.__name__
        if wrapped_name not in cls.registry():
            cls.registry().register(name=wrapped_name, obj=cls)

    def __init__(self, *args, **kwargs):
        if kwargs:
            logger.warning(f"Unused kwargs: {kwargs} in {self.__class__.__name__}")

    @staticmethod
    def base_class_adapter(external_cls: type, base_cls: type) -> type:
        """Decorator to make any external class inherit from BaseClass"""

        class WrappedClass(external_cls, base_cls, wrapped_name=external_cls.__name__):
            def __init__(self, *args, **kwargs):
                external_cls.__init__(self, *args, **kwargs)
                base_cls.__init__(self, *args, **kwargs)

        return WrappedClass


from . import data, learning, models, tasks, trainer

__all__ = ["data", "learning", "models", "tasks", "trainer"]
