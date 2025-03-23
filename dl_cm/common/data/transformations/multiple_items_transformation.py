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
    
    def __init__(self, parent_transformation : GeneralRevrsibleTransformation,
                 included_keys:Optional[Union[Iterable,Callable]]=None,
                 ignored_keys:Optional[Union[Iterable,Callable]]=None):
        self.parent_transformation=parent_transformation
        
        if bool(included_keys) and bool(ignored_keys):
            raise Exception("Include and exclude arguments are provided at the same time")
        
        if callable(included_keys):
            self.included_keys=included_keys
        elif is_iterable(included_keys):
            included_keys = set(included_keys)
            self.included_keys = lambda x: x in included_keys
        elif included_keys is None:
            if ignored_keys is None:
                self.included_keys = lambda x:True
            elif callable(ignored_keys):
                self.included_keys = lambda x: not ignored_keys(v)
            elif is_iterable(ignored_keys):
                ignored_keys = set(ignored_keys)
                self.included_keys = lambda x: x not in ignored_keys
            else:
                raise OutOfTypesException(ignored_keys, (Callable, Iterable))
        else:
            raise OutOfTypesException(included_keys, (Callable, Iterable))
                
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