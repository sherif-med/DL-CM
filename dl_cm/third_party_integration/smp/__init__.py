
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, JaccardLoss, SoftBCEWithLogitsLoss
from dl_cm.common.tasks.criterion import CRITIREON_REGISTRY, base_loss_adapter

CRITIREON_REGISTRY.register(
    obj=SoftCrossEntropyLoss, name="SoftCrossEntropyLoss", base_class_adapter=base_loss_adapter
)

CRITIREON_REGISTRY.register(
    obj=SoftBCEWithLogitsLoss, name="SoftBCEWithLogitsLoss", base_class_adapter=base_loss_adapter
)
