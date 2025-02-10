from dl_cm.common.learning.criterion_learner import CriterionLearner
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
from dl_cm.common.typing import lossOutputStruct, StepInputStruct, StepOutputStruct

class UnsupervisedLearner(CriterionLearner, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            predicted_key: str
        return ValidConfig

    def __init__(self, learner_config:dict):
        validationMixin.__init__(self, learner_config)
        CriterionLearner.__init__(self, learner_config)
        self.predicted_key : str = learner_config.pop("predicted_key")
    
    def step(self, batch: StepInputStruct, compute_loss=True) -> StepOutputStruct:
        predictions = self.model()(batch["inputs"])
        if compute_loss:
            loss_dict = self.criteron_step(predictions)
        else:
            loss_dict = {}
        output = {"predictions":predictions, "loss":loss_dict}
        return output

    def criteron_step(self, predictions:dict) -> lossOutputStruct:
        predicted = predictions[self.predicted_key]
        loss = self.criterion(predicted)
        return loss