from dl_cm.common.learning.criterion_learner import CriterionLearner
from dl_cm.common.typing import StepInputStruct, StepOutputStruct, lossOutputStruct


class SupervisedLearner(CriterionLearner):
    def __init__(self, target_key: str, predicted_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_key: str = target_key
        self.predicted_key: str = predicted_key

    def forward(self, batch: StepInputStruct, compute_loss=True) -> StepOutputStruct:
        predictions = self.model(batch["inputs"])
        if compute_loss:
            targets = batch["targets"]
            loss_struct = self.criteron_step(targets, predictions)
            loss_value = loss_struct.value()
        else:
            loss_struct = {}
            loss_value = None
        output = {"predictions": predictions, "loss": loss_value, "losses": loss_struct}
        return output

    def criteron_step(self, targets: dict, predictions: dict) -> lossOutputStruct:
        target = targets[self.target_key]
        predicted = predictions[self.predicted_key]
        loss_dict = self.criterion.forward(predicted, target)
        return loss_dict
