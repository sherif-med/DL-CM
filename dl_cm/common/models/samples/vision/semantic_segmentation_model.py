from dl_cm.common.models.base_model import BaseModel
import segmentation_models_pytorch as smp

class SemanticSegmentationModel(BaseModel):
    def __init__(self, model_config:dict):
        super().__init__(model_config)
        self.model = smp.create_model(model_config)   
    
    def forward(self, input):
        seg_map = self.model(input)
        return {"seg_map": seg_map}