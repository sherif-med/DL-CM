import torch
import torch.nn as nn
from torchmetrics import MeanMetric, MetricCollection

from dl_cm import _logger as logger
from dl_cm.common import DLCM
from dl_cm.common.typing import lossOutputStruct, namedEntitySchema
from dl_cm.utils.exceptions import OutOfTypesException
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.utils.ppattern.named_mixin import NamedMixin
from dl_cm.utils.registery import Registry

CRITIREON_REGISTRY = Registry("Critireon")


def base_loss_adapter(loss_cls: type[nn.modules.loss._Loss]):
    """Decorator to make any external loss inherit from BaseLoss
    and adapt output to lossOutputStruct
    """

    class WrappedLoss(loss_cls, BaseLoss):
        def __init__(self, *args, **kwargs):
            """Wraps a loss to extract values from dictionary inputs."""
            loss_cls.__init__(self, *args, **kwargs)
            BaseLoss.__init__(self, *args, **kwargs)

        @staticmethod
        def adapt_output_struct(output_loss):
            if isinstance(output_loss, lossOutputStruct):
                return output_loss
            if isinstance(output_loss, (torch.Tensor, float)):
                output_loss = lossOutputStruct(
                    name=loss_cls.__name__, losses={loss_cls.__name__: output_loss}
                )
                return output_loss
            if isinstance(output_loss, dict):
                if not loss_cls.__name__ in output_loss:
                    logger.critical(f"Metric {loss_cls.__name__} is not in loss dict!")
                output_loss = lossOutputStruct(
                    name=loss_cls.__name__, losses=output_loss
                )
                return output_loss
            raise OutOfTypesException(
                output_loss, (lossOutputStruct, torch.Tensor, float, dict)
            )

        def forward(self, preds: dict, target: dict = None):
            """Extract tensors and call the original update method."""
            pred_tensor = preds[self.preds_key] if self.preds_key else preds
            if target is None:
                loss_output = super().forward(pred_tensor)
                return self.adapt_output_struct(loss_output)
            target_tensor = target[self.target_key] if self.target_key else target
            loss_output = super().forward(pred_tensor, target_tensor)
            return self.adapt_output_struct(loss_output)

    return WrappedLoss


class BaseLoss(NamedMixin, DLCM):
    @staticmethod
    def registry() -> Registry:
        return CRITIREON_REGISTRY

    def __init__(self, preds_key: str = None, target_key: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds_key = preds_key
        self.target_key = target_key

    def as_metric_collection(self) -> MetricCollection:
        return MetricCollection({self.name(): MeanMetric()})


class CombinedLoss(BaseLoss):
    def __init__(
        self,
        losses: list[str, namedEntitySchema, BaseLoss],
        weights: list[float] = None,
        *args,
        **kwargs,
    ):
        """
        :param losses: A list of losses to be combined. Each loss can be either a string
            (in which case it is a key in the CRITIREON_REGISTRY), a dictionary (in which
            case it is passed to the CritireonFactory to construct a loss), or an instance
            of nn.modules.loss._Loss.
        :param weights: A list of weights to be used in the combination. If None, the
            weights will be set to equal values (i.e. the losses will be equally weighted).
            If the length of the weights list is not equal to the length of the losses list,
            the weights will be set to equal values.
        :return:
        """

        super().__init__(*args, **kwargs)
        losses = CritireonFactory.create(losses)
        self.losses: list[BaseLoss] = nn.ModuleList(losses)

        if weights is None:
            logger.info("Defaulting to equal weighting of losses")
            weights = [1.0] * len(losses)
        if len(weights) != len(losses) or any({w is None for w in weights}):
            logger.warning(
                "Number of weights does not match number of losses. Defaulting to equal weighting of losses"
            )
            weights = [1.0] * len(losses)

        self.weights = weights
        if sum(self.weights) != 1:
            self.weights = list(map(lambda x: x / sum(self.weights), self.weights))

    def forward(self, prediction, target) -> lossOutputStruct:
        losses_dict = {}
        total_loss = 0.0
        # Iterate over each loss function and corresponding weight
        for loss_fn, weight in zip(self.losses, self.weights):
            c_loss_value = loss_fn(prediction, target)
            losses_dict[loss_fn.name()] = c_loss_value
            total_loss += c_loss_value * weight
        losses_dict[self.name()] = total_loss
        return lossOutputStruct(name=self.name(), losses=losses_dict)

    def as_metric_collection(self):
        # Adding current loss to the metric collection
        metric_collection = MetricCollection({self.name(): MeanMetric()})
        # Adding child losses to the metric collection
        loss_fn: BaseLoss
        for loss_fn in self.losses:
            metric_collection.add_metrics(loss_fn.as_metric_collection())
        return metric_collection


class CritireonFactory(BaseFactory[BaseLoss]):
    @staticmethod
    def base_class() -> type[BaseLoss]:
        return BaseLoss


for name in dir(nn.modules.loss):
    attr = getattr(nn.modules.loss, name)
    if isinstance(attr, type) and issubclass(attr, nn.modules.loss._Loss):
        CRITIREON_REGISTRY.register(
            obj=attr, name=name, base_class_adapter=base_loss_adapter
        )
