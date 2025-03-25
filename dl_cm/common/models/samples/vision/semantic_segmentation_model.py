import pydantic as pd
import segmentation_models_pytorch as smp
import torch

from dl_cm.common.models import BaseModel


class SemanticSegmentationModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = smp.create_model(*args, **kwargs)

    @classmethod
    def get_prediction_schema(cls) -> pd.BaseModel:
        class PredictionSchema(pd.BaseModel):
            seg_map: torch.Tensor

        return PredictionSchema

    def forward(self, input):
        seg_map = self.model(input)
        return {"seg_map": seg_map}
