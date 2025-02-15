from dl_cm.utils.ppattern.self_registered_mixin import selfRegisteredMixin

class DLCM(selfRegisteredMixin):

    @staticmethod
    def base_class_adapter(external_cls: type, base_cls: type)-> type:
        """Decorator to make any external class inherit from BaseClass"""
        class WrappedClass(external_cls, base_cls):
            def __init__(self, config:dict):
                base_cls.__init__(self, config)
                external_cls.__init__(**config.get("params", {}))
        return WrappedClass