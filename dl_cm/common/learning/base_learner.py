from dl_cm.common.tasks.base_task import BaseTask

class BaseLearner(BaseTask):
    def __init__(self, config) -> None:
        self.config = config