
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, JaccardLoss
from dl_cm.common.tasks.criterion import CRITIREON_REGISTRY, base_loss_adapter

CRITIREON_REGISTRY.register(
    obj=DiceLoss, name="DiceLoss", base_class_adapter=base_loss_adapter
)

CRITIREON_REGISTRY.register(
    obj=SoftCrossEntropyLoss, name="SoftCrossEntropyLoss", base_class_adapter=base_loss_adapter
)

CRITIREON_REGISTRY.register(
    obj=JaccardLoss, name="JaccardLoss", base_class_adapter=base_loss_adapter
)