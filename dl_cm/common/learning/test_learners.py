from dl_cm.common.learning.equivariance_enforced_learner import equivarianceEnforcedLearner
from dl_cm.common.typing import lossOutputStruct, StepInputStruct, StepOutputStruct
from dl_cm.common.tasks.criterion import BaseLoss
import torch
from dl_cm import _logger as logger

class acrossDimensionVarianceLoss(BaseLoss):

    def __init__(self, across_dimension : int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._across_dimension = across_dimension

    def forward(self, prediction: torch.Tensor) -> lossOutputStruct:
        std_value = torch.std(prediction.flatten(1), dim=self._across_dimension).mean()
        return lossOutputStruct(name=self.name(), losses={self.name():std_value})

class SegmentationEquivarianceEnforcedLearner(equivarianceEnforcedLearner):

    def __init__(self, segmentation_key: str, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.segmentation_key = segmentation_key

    def criteron_step(self, predictions: dict) -> lossOutputStruct:
        if self.segmentation_key not in predictions.keys():
            logger.error(f"Segmentation key {self.segmentation_key} not found in predictions")
        else:
            return super().criteron_step(predictions[self.segmentation_key])