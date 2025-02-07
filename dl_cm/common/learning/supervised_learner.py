from dl_cm.common.learning.criterion_learner import CriterionLearner
from dl_cm.utils.ppattern.data_validation import validationMixin
import pydantic as pd
from dl_cm.common.tasks.base_task import StepOutputStruct, StepInputStruct
from dl_cm.common.tasks.criterion import lossOutputStruct

class SupervisedLearner(CriterionLearner, validationMixin):

    def config_schema(cls)-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            target_key: str
            predicted_key: str
        return ValidConfig

    def __init__(self, learner_config:dict):
        validationMixin.__init__(self, learner_config)
        CriterionLearner.__init__(self, learner_config)
        self.target_key : str = learner_config.pop("target_key")
        self.predicted_key : str = learner_config.pop("predicted_key")
    
    def step(self, batch: StepInputStruct, compute_loss=True) -> StepOutputStruct:
        predictions = self.model()(batch["inputs"])
        if compute_loss:
            targets = batch["targets"]
            loss_dict = self.criteron_step(targets, predictions)
        else:
            loss_dict = {}
        output = {"predictions":predictions, "loss":loss_dict}
        return output

    def criteron_step(self, targets:dict, predictions:dict) -> lossOutputStruct:
        target = targets[self.target_key]
        predicted = predictions[self.predicted_key]
        loss_dict = self.criterion(predicted, target)
        return loss_dict