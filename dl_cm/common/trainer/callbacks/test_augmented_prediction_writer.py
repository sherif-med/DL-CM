
from lightning import LightningModule, Trainer
from skimage.io import imsave
from dl_cm.common.trainers.callbacks.prediction_writer import PostPredictionCallback
from collections.abc import Iterable, Callable
from typing import Dict
from dl_cm.common.datasets.transformations.general_transformation import GeneralRevrsibleTransformation
from dl_cm.utils.ppattern.batch_operations import merge_multiple_batches, split_into_batches, get_batch_size_from_items_dict

class TestAUgmentedPredictionWriter(PostPredictionCallback):

    #@rank_zero_only
    def __init__(
        self,
        output_dir,
        write_interval,
        preds_items_callback,
        transfromations: Iterable[GeneralRevrsibleTransformation],
        aggregate_prediction: Dict[str, Callable]
    ):
        super().__init__(output_dir, write_interval, preds_items_callback)
        self.transfromations = list(transfromations)
        self.aggregate_prediction = aggregate_prediction

    def on_predict_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        ### Apply augmentation before inference through model
        transformed_batches = [t(batch) for t in self.transfromations]
        merged_batches = merge_multiple_batches(transformed_batches)
        batch.update(merged_batches)
        return super().on_predict_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
    
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        ## AGgregate augmentations after inference && call super on_predict_batch_end (todo check if must be done)
        
        augmented_batch_size = get_batch_size_from_items_dict(batch)
        
        initial_batch_size = augmented_batch_size // len(self.transfromations)
        
        #Updating batch
        split_batch = split_into_batches(batch, initial_batch_size)
        batch.update(split_batch[0])
        
        split_prediction = split_into_batches(outputs["preds"], initial_batch_size)
        
        reverse_transformed_prediction = []
        
        for c_prediction, c_respective_transform in zip(split_prediction, self.transfromations):
            reverse_transformed_prediction.append(
                c_respective_transform(c_prediction, reverse=True)
            )
        
        for pred_name, agg_fn in self.aggregate_prediction.items():
            outputs["preds"][pred_name] = agg_fn(
                [pred[pred_name] for pred in reverse_transformed_prediction]
            )
                
        return super().on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)