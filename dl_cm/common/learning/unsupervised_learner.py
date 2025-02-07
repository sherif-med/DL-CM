from dl_cm.common.learning.criterion_learner import CriterionLearner
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
from dl_cm.common.tasks.criterion import lossOutputStruct

class UnsupervisedLearner(CriterionLearner, validationMixin):

    def config_schema(cls)-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            predicted_key: str
        return ValidConfig

    def __init__(self, learner_config:dict):
        validationMixin.__init__(self, learner_config)
        CriterionLearner.__init__(self, learner_config)
        self.predicted_key : str = learner_config.pop("predicted_key")
    
    def criteron_step(self, predictions:dict) -> lossOutputStruct:
        predicted = predictions[self.predicted_key]
        loss = self.criterion(predicted)
        return loss