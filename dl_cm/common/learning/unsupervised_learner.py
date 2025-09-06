from dl_cm.common.learning.criterion_learner import CriterionLearner
from dl_cm.common.typing import StepInputStruct, StepOutputStruct, lossOutputStruct


class UnsupervisedLearner(CriterionLearner):
    def __init__(self, predicted_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predicted_key: str = predicted_key

    def forward(self, batch: StepInputStruct, compute_loss=True) -> StepOutputStruct:
        predictions = self.model()(batch["inputs"])
        if compute_loss:
            loss_struct = self.criteron_step(predictions)
            loss_value = loss_struct.value()
        else:
            loss_struct = {}
            loss_value = None
        output = {"predictions": predictions, "loss": loss_value, "losses":loss_struct}
        return output

    def criteron_step(self, predictions: dict) -> lossOutputStruct:
        predicted = predictions[self.predicted_key]
        loss = self.criterion(predicted)
        return loss
