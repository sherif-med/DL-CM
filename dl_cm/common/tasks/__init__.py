from dl_cm.utils.registery import Registry
from .base_task import BaseTask
from dl_cm.utils.ppattern.factory import BaseFactory

TASKS_REGISTERY = Registry("Tasks")

class TasksFactory(BaseFactory):

    @staticmethod
    def base_class()-> type:
        return BaseTask
    