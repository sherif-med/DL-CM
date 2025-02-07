from dl_cm.common.tasks.base_task import StepInputStruct, StepOutputStruct

class BaseLearner:
    def __init__(self, config) -> None:
        self.config = config
    
    def model(self):
        raise NotImplementedError

    def step(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        raise NotImplementedError