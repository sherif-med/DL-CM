from dl_cm.common.learning.base_learner import BaseLearner
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
from dl_cm.common.learning import LearnersFactory
from dl_cm.common.typing import StepOutputStruct, StepInputStruct

class MultiLearner(BaseLearner, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            learners: list[dict]
        return ValidConfig
    
    def __init__(self, config):
        validationMixin.__init__(self, config)
        BaseLearner.__init__(config)
        self.learners : list[BaseLearner] = LearnersFactory.create(config.get("learners"))
    
    def aggregate_output(self, outputs: list[StepOutputStruct]) -> StepOutputStruct:
        raise NotImplementedError
    
    def step(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        leraners_output : list[StepOutputStruct] = []
        for learner in self.learners:
            leraners_output.append(learner.step(batch, *args, **kwargs))
        return self.aggregate_output(leraners_output)

class SequentialLearner(BaseLearner, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            learners: list[dict]
        return ValidConfig
    
    def __init__(self, config):
        validationMixin.__init__(self, config)
        BaseLearner.__init__(config)
        self.learners : list[BaseLearner] = LearnersFactory.create(config.get("learners"))
    
    def step(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        for learner in self.learners:
            batch = learner.step(batch, *args, **kwargs)
        return batch

class learnerWrapper(BaseLearner, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            wraped_learner: dict
        return ValidConfig

    def __init__(self, config):
        validationMixin.__init__(self, config)
        BaseLearner.__init__(config)
        self.wraped_learner : BaseLearner = LearnersFactory.create(config.get("wraped_learner"))
    
    def pre_step(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        raise NotImplementedError
    
    def post_step(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        raise NotImplementedError

    def step(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        batch = self.pre_step(batch, *args, **kwargs)
        batch = self.wraped_learner.step(batch, *args, **kwargs)
        batch = self.post_step(batch, *args, **kwargs)
        return batch
