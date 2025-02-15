from dl_cm.utils.ppattern.batch_operations import merge_multiple_batches, split_into_batches, get_batch_size_from_items_dict
from dl_cm.common.datasets.transformations.general_transformation import GeneralRevrsibleTransformation, GeneralTransformationFactory
from dl_cm.common.learning.unsupervised_learner import UnsupervisedLearner
from dl_cm.common.learning.extra_learners import learnerWrapper
from dl_cm.utils.ppattern.data_validation import validationMixin
from dl_cm.common.typing import StepOutputStruct, StepInputStruct
from dl_cm.common.typing import lossOutputStruct
from dl_cm import _logger as logger
import pydantic as pd

class equivarianceEnforcedLearner(learnerWrapper, UnsupervisedLearner, validationMixin):

    @staticmethod
    def config_schema()-> pd.BaseModel:
        class ValidConfig(pd.BaseModel):
            input_transformations: list
            output_transformations: list = None
        return ValidConfig

    def __init__(self, config):
        validationMixin.__init__(self, config)
        learnerWrapper.__init__(self, config)
        UnsupervisedLearner.__init__(self, config)
        self.input_transformations_callable : list[GeneralRevrsibleTransformation] = \
            [GeneralTransformationFactory.create(t) for t in config.get("input_transformations")]
        if config.get("output_transformations") is not None:
            self.output_transformations_callable : list[GeneralRevrsibleTransformation] = \
                [GeneralTransformationFactory.create(t) for t in config.get("output_transformations")]
        else:
            self.output_transformations_callable = self.input_transformations_callable

    def pre_step(self, batch: StepInputStruct, *args, **kwargs) -> StepInputStruct:
        """
        Apply transformations to the input batch and merge the results into a single batch.
        The order of transformations is the same as the order of input_transformations in the config.
        The output of this function is a StepInputStruct which is the input for the underlying learner.
        The output is used as the input for the underlying learner.
        """
        transformed_batches = [t(batch) for t in self.input_transformations_callable]
        merged_batches = merge_multiple_batches(transformed_batches)
        return merged_batches

    def post_step(self, batch: StepOutputStruct, *args, **kwargs) -> StepOutputStruct:
        """
        Perform post-processing on the output batch by reversing transformations and computing losses and predictions.

        This function splits the prediction output into batches, applies the reverse of the transformations used during
        pre-processing, and computes the relevant losses and aggregated predictions.
        """

        split_prediction : list[dict] = split_into_batches(batch["predictions"], get_batch_size_from_items_dict(batch))
        reverse_transformed_prediction = []
        for c_prediction, c_respective_transform in zip(split_prediction, self.output_transformations_callable):
            reverse_transformed_prediction.append(
                c_respective_transform(c_prediction, reverse=True)
            )

        var_loss : lossOutputStruct = self.criteron_step(reverse_transformed_prediction)
        if (var_loss.losses.keys() & batch.loss.losses.keys()):
            logger.warning(f"Conflicting loss keys {list(var_loss.keys() & batch.loss.keys())}!")
            exit(1)

        # adding variance reduction loss to the batch loss and updating the losses dict
        aggregated_var_losses = var_loss.losses[var_loss.name()]
        batch.loss.losses[batch.loss.name()] += aggregated_var_losses # TODO change aggregation of losses
        batch.loss.losses |= var_loss.losses

        return batch
