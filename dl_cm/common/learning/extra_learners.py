from dl_cm.common.learning.base_learner import BaseLearner
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
from dl_cm.common.learning import LearnersFactory
from dl_cm.common.typing import StepOutputStruct, StepInputStruct
from abc import ABC, abstractmethod

class MultiLearner(BaseLearner, validationMixin, ABC):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            learners: list[dict]
        return ValidConfig
    
    def __init__(self, config):
        validationMixin.__init__(self, config)
        BaseLearner.__init__(config)
        self.learners : list[BaseLearner] = LearnersFactory.create(config.get("learners"))
    
    @abstractmethod
    def aggregate_output(self, outputs: list[StepOutputStruct]) -> StepOutputStruct:
        pass
    
    def forward(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        leraners_output : list[StepOutputStruct] = []
        for learner in self.learners:
            leraners_output.append(learner.forward(batch, *args, **kwargs))
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
    
    def forward(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        for learner in self.learners:
            batch = learner.forward(batch, *args, **kwargs)
        return batch

class learnerWrapper(BaseLearner, validationMixin, ABC):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            wraped_learner: dict
        return ValidConfig

    def __init__(self, config):
        validationMixin.__init__(self, config)
        BaseLearner.__init__(config)
        self.wraped_learner : BaseLearner = LearnersFactory.create(config.get("wraped_learner"))
    
    @abstractmethod
    def pre_step(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        pass
    
    @abstractmethod
    def post_step(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        pass

    def forward(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        batch = self.pre_step(batch, *args, **kwargs)
        batch = self.wraped_learner.forward(batch, *args, **kwargs)
        batch = self.post_step(batch, *args, **kwargs)
        return batch
