from dl_cm.common.learning.criterion_learner import CriterionLearner
from dl_cm.common.typing import StepInputStruct, StepOutputStruct, lossOutputStruct


class UnsupervisedLearner(CriterionLearner):
    def __init__(self, predicted_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predicted_key: str = predicted_key

    def forward(self, batch: StepInputStruct, compute_loss=True) -> StepOutputStruct:
        predictions = self.model()(batch["inputs"])
        if compute_loss:
            loss_dict = self.criteron_step(predictions)
        else:
            loss_dict = {}
        output = {"predictions": predictions, "loss": loss_dict}
        return output

    def criteron_step(self, predictions: dict) -> lossOutputStruct:
        predicted = predictions[self.predicted_key]
        loss = self.criterion(predicted)
        return loss
