from dl_cm.utils.exceptions import BaseClassNotInitializedException
class InitCheckMixin:
    """Mixin to check if a base class has been initialized."""
    _initialized = False  # Class-level flag to track initialization

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Continue MRO chain
        # Mark this mixin's initialization as complete
        self.__class__._initialized = True
    
    @property
    def base_class_initialized(self):
        return self.__class__._initialized
    
    def check_base_class_initialized(self):
        """
        Raise an exception if the base class has not been initialized.
        This method should be called in methods that require the base class to be initialized.
        """
        if not self.base_class_initialized:
            raise BaseClassNotInitializedException(self)