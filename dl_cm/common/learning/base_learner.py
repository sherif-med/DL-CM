from dl_cm.common.typing import StepInputStruct, StepOutputStruct
    def __init__(self, config) -> None:
        self.config = config
    
    def model(self):
        raise NotImplementedError

    def step(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        raise NotImplementedError