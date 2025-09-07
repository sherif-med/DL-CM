import pydantic as pd
import segmentation_models_pytorch as smp
import torch

from dl_cm.common.models import BaseModel


class SemanticSegmentationModel(BaseModel):
    def __init__(self, input_key, seg_output_key="seg_map", label_output_key="label_map", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = smp.create_model(*args, **kwargs)
        self.input_key = input_key
        self.seg_output_key = seg_output_key
        self.label_output_key = label_output_key


    @classmethod
    def get_prediction_schema(cls) -> pd.BaseModel:
        class PredictionSchema(pd.BaseModel):
            seg_map: torch.Tensor

        return PredictionSchema

    def forward(self, input):
        seg_map = self.model(input[self.input_key])
        label_map = torch.softmax(seg_map,1).argmax(dim=1)
        return {self.seg_output_key: seg_map, self.label_output_key:label_map}
