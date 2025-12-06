from .general_transformation import GeneralRevrsibleTransformation
import copy
from dl_cm.utils.exceptions import OutOfTypesException
from typing import Union, Optional
from collections.abc import Iterable, Callable

def is_iterable(v):
    try:
        iterator = iter(v)
        return True
    except TypeError:
        return False

class MultipleItemRevrsibleTransformation(GeneralRevrsibleTransformation):
    
    def __init__(self, parent_transformation: GeneralRevrsibleTransformation,
                 included_keys: Optional[Union[Iterable, Callable]] = None,
                 ignored_keys: Optional[Union[Iterable, Callable]] = None):
        self.parent_transformation = parent_transformation
        
        if included_keys is not None and ignored_keys is not None:
            raise Exception("Include and exclude arguments are provided at the same time")
        
        # Store the original data instead of closures
        self._included_callable = None
        self._included_set = None
        self._ignored_callable = None
        self._ignored_set = None
        self._always_include = False
        
        if callable(included_keys):
            self._included_callable = included_keys
        elif is_iterable(included_keys):
            self._included_set = set(included_keys)
        elif included_keys is None:
            if ignored_keys is None:
                self._always_include = True
            elif callable(ignored_keys):
                self._ignored_callable = ignored_keys
            elif is_iterable(ignored_keys):
                self._ignored_set = set(ignored_keys)
            else:
                raise OutOfTypesException(ignored_keys, (Callable, Iterable))
        else:
            raise OutOfTypesException(included_keys, (Callable, Iterable))
    
    def included_keys(self, x):
        """Check if key should be included"""
        if self._always_include:
            return True
        if self._included_callable:
            return self._included_callable(x)
        if self._included_set is not None:
            return x in self._included_set
        if self._ignored_callable:
            return not self._ignored_callable(x)
        if self._ignored_set is not None:
            return x not in self._ignored_set
        return True
        
    def __fwd__(self, items_dict:dict):
        out_items_dict = {}
        for k,v in items_dict.items():
            if self.included_keys(k):
                out_items_dict[k] = self.parent_transformation.__fwd__(v)
            else:
                out_items_dict[k] = copy.deepcopy(v)
        return out_items_dict
    def __rwd__(self, items_dict):
        out_items_dict = {}
        for k,v in items_dict.items():
            if self.included_keys(k):
                out_items_dict[k] = self.parent_transformation.__rwd__(v)
            else:
                out_items_dict[k] = copy.deepcopy(v)
        return out_items_dict