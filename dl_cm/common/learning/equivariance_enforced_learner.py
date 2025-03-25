from dl_cm.common.data.transformations.general_transformation import (
    GeneralRevrsibleTransformation,
    GeneralTransformation,
    GeneralTransformationFactory,
)
from dl_cm.common.learning.extra_learners import learnerWrapper
from dl_cm.common.learning.unsupervised_learner import UnsupervisedLearner
from dl_cm.common.typing import (
    StepInputStruct,
    StepOutputStruct,
    lossOutputStruct,
    namedEntitySchema,
)
from dl_cm.utils.ppattern.batch_operations import (
    get_batch_size_from_items_dict,
    merge_multiple_batches,
    split_into_batches,
)


class equivarianceEnforcedLearner(learnerWrapper, UnsupervisedLearner):
    def __init__(
        self,
        input_transformations: list[
            GeneralTransformation | GeneralRevrsibleTransformation | namedEntitySchema
        ],
        output_transformations: list[GeneralTransformation] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # UnsupervisedLearner.__init__(self, config)
        self.input_transformations_callable: list[GeneralTransformation] = [
            GeneralTransformationFactory.create(t) for t in input_transformations
        ]
        if output_transformations:
            self.output_transformations_callable: list[GeneralTransformation] = [
                GeneralTransformationFactory.create(t) for t in output_transformations
            ]
        else:
            self.output_transformations_callable = [
                ~t for t in self.input_transformations_callable
            ]

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

        split_prediction: list[dict] = split_into_batches(
            batch["predictions"], get_batch_size_from_items_dict(batch)
        )
        reverse_transformed_prediction = []
        for c_prediction, c_respective_transform in zip(
            split_prediction, self.output_transformations_callable
        ):
            reverse_transformed_prediction.append(c_respective_transform(c_prediction))

        var_loss: lossOutputStruct = self.criteron_step(reverse_transformed_prediction)
        if var_loss.losses.keys() & batch.loss.losses.keys():
            raise RuntimeError(
                f"Conflicting loss keys {list(var_loss.losses.keys() & batch.loss.losses.keys())}!"
            )

        # adding variance reduction loss to the batch loss and updating the losses dict
        aggregated_var_losses = var_loss.losses[var_loss.name()]
        batch.loss.losses[batch.loss.name()] += (
            aggregated_var_losses  # TODO change aggregation of losses
        )
        batch.loss.losses |= var_loss.losses

        return batch
