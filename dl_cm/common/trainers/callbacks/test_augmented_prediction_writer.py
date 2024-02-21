

from pathlib import Path
from pytorch_lightning import LightningModule, Trainer
from skimage.io import imsave
from dl_cm.common.trainers.callbacks.prediction_writer import PostPredictionCallback
import numpy as np
from collections.abc import Iterable, Callable
from typing import Dict
from dl_cm.common.datasets.transformations.general_transformation import GeneralRevrsibleTransformation
import torch
import itertools

def merge_multiple_batches(batches):
    merged_dict = {}
    for key, value in batches[0].items():
        # Concatenate tensors from all dictionaries for each key
        if isinstance(value, torch.Tensor):
            merged_dict[key] = torch.cat([batch[key] for batch in batches], dim=0)
        else:
            merged_dict[key]=list(itertools.chain(*[batch[key] for batch in batches]))
    return merged_dict

def split_into_batches(merged_dict, augmented_batch_size, initial_batch_size):
    # Calculate the total number of batches
    total_batches = augmented_batch_size // initial_batch_size    
    # Initialize a list to hold the split dictionaries (batches)
    batches = []    
    for i in range(total_batches):
        # Calculate the start and end indices for the current batch
        start_idx = i * initial_batch_size
        end_idx = start_idx + initial_batch_size        
        # Create a new dictionary for the current batch by slicing each tensor
        batch_dict = {key: tensor[start_idx:end_idx] for key, tensor in merged_dict.items()}        
        # Add the current batch dictionary to the list
        batches.append(batch_dict)    
    return batches

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
        
        def get_batch_size_from_items_dict(items_dict):
            for k,v in items_dict.items():
                if isinstance(v, torch.Tensor):
                    return v.size(0)
            raise Exception("Unable to get batch size!")
        
        augmented_batch_size = get_batch_size_from_items_dict(batch)
        
        initial_batch_size = augmented_batch_size // len(self.transfromations)
        
        #Updating batch
        split_batch = split_into_batches(batch, augmented_batch_size, initial_batch_size)
        batch.update(split_batch[0])
        
        split_prediction = split_into_batches(outputs["preds"], augmented_batch_size, initial_batch_size)
        
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