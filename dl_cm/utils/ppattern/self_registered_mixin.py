from dl_cm.utils.registery import registeredClassMixin 

class selfRegisteredMixin(registeredClassMixin):

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.registry().register(cls.__name__, cls)